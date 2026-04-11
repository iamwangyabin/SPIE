import copy
from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.vit_tunamax import VisionTransformer as TunaMaxVisionTransformer


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.scale = float(alpha) / float(rank)
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.up_proj = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return self.up_proj(self.down_proj(x)) * self.scale


class MLPLoRAAdapter(nn.Module):
    """MLP LoRA adapter branch for one token segment."""

    def __init__(self, dim, mlp_hidden_dim, rank=8, alpha=1.0):
        super().__init__()
        self.fc1_lora = LoRALinear(dim, mlp_hidden_dim, rank=rank, alpha=alpha)
        self.fc2_lora = LoRALinear(mlp_hidden_dim, dim, rank=rank, alpha=alpha)
        self.up_proj = self.fc2_lora.up_proj


class CaSSLePredictor(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = int(hidden_dim or dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class VeraLinear(nn.Module):
    """VeRA delta branch with trainable lambda_d/lambda_b only."""

    def __init__(self, in_features, out_features, rank=256, dropout=0.0, d_initial=0.1):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lambda_d = nn.Parameter(torch.full((self.rank,), float(d_initial)))
        self.lambda_b = nn.Parameter(torch.zeros(self.out_features))

    def forward(self, x, shared_proj):
        a = shared_proj.A[:, : self.in_features]
        b = shared_proj.B[: self.out_features, :]

        x_adapt = self.dropout(x).to(self.lambda_d.dtype)
        a = a.to(device=x_adapt.device, dtype=x_adapt.dtype)
        b = b.to(device=x_adapt.device, dtype=x_adapt.dtype)

        z = F.linear(x_adapt, a)
        z = z * self.lambda_d
        delta = F.linear(z, b)
        delta = delta * self.lambda_b
        return delta.to(x.dtype)


class MLPVeRAAdapter(nn.Module):
    """Expert MLP adapter with VeRA updates on fc1/fc2."""

    def __init__(self, dim, mlp_hidden_dim, rank=256, dropout=0.0, d_initial=0.1):
        super().__init__()
        self.fc1_lora = VeraLinear(dim, mlp_hidden_dim, rank=rank, dropout=dropout, d_initial=d_initial)
        self.fc2_lora = VeraLinear(mlp_hidden_dim, dim, rank=rank, dropout=dropout, d_initial=d_initial)


class FrozenVeraProjection(nn.Module):
    """Frozen VeRA projection pair derived from a pretrained linear weight."""

    def __init__(self, a, b, save_projection=True):
        super().__init__()
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("VeRA projection tensors must be rank-2.")
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"Projection rank mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")

        self.register_buffer("A", a, persistent=bool(save_projection))
        self.register_buffer("B", b, persistent=bool(save_projection))
        self.r = int(a.shape[0])

    @classmethod
    def from_linear_weight(cls, weight, rank=256, save_projection=True):
        if weight.ndim != 2:
            raise ValueError(f"Expected a rank-2 linear weight, got shape {tuple(weight.shape)}")

        out_features, in_features = weight.shape
        rank = int(rank)
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        with torch.no_grad():
            weight_fp32 = weight.detach().to(dtype=torch.float32)
            u, _, vh = torch.linalg.svd(weight_fp32, full_matrices=False)

            effective_rank = min(rank, vh.shape[0], u.shape[1])
            a = weight_fp32.new_zeros((rank, in_features))
            b = weight_fp32.new_zeros((out_features, rank))
            if effective_rank > 0:
                a[:effective_rank] = vh[:effective_rank]
                b[:, :effective_rank] = u[:, :effective_rank]

        return cls(a=a.to(dtype=weight.dtype), b=b.to(dtype=weight.dtype), save_projection=save_projection)

    def extra_repr(self):
        return f"A={tuple(self.A.shape)}, B={tuple(self.B.shape)}, r={self.r}"


class MLPMixedAdapterBlock(nn.Module):
    """TunaMax block wrapper with shared-LoRA and expert-VeRA MLP branches."""

    def __init__(self, base_block):
        super().__init__()
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.drop_path = base_block.drop_path
        self.norm2 = base_block.norm2
        self.fc1 = base_block.fc1
        self.fc2 = base_block.fc2
        self.act = base_block.act
        self.mlp_drop = base_block.mlp_drop

    def _forward_attention(self, x, attn_mask=None):
        if attn_mask is None:
            return self.attn(x)

        bsz, seq_len, dim = x.shape
        num_heads = self.attn.num_heads
        head_dim = self.attn.head_dim

        q = self.attn.q_proj(x)
        k = self.attn.k_proj(x)
        v = self.attn.v_proj(x)

        q = self.attn._shape(q, seq_len, bsz).view(bsz * num_heads, seq_len, head_dim)
        k = self.attn._shape(k, -1, bsz).view(bsz * num_heads, -1, head_dim)
        v = self.attn._shape(v, -1, bsz).view(bsz * num_heads, -1, head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.attn.scale
        attn_weights = attn_weights.view(bsz, num_heads, seq_len, seq_len)
        attn_weights = attn_weights + attn_mask.to(device=attn_weights.device, dtype=attn_weights.dtype).view(
            1, 1, seq_len, seq_len
        )
        attn_weights = attn_weights.view(bsz * num_heads, seq_len, seq_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn.attn_drop(attn_probs)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(bsz, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, dim)
        attn_output = self.attn.proj(attn_output)
        attn_output = self.attn.proj_drop(attn_output)
        return attn_output

    def forward(
        self,
        x,
        shared_lora=None,
        expert_lora=None,
        attn_mask=None,
        num_backbone_tokens=None,
        expert_fc1_proj=None,
        expert_fc2_proj=None,
    ):
        x = x + self.drop_path(self._forward_attention(self.norm1(x), attn_mask=attn_mask))

        residual = x
        norm2_x = self.norm2(x)
        hidden = self.fc1(norm2_x)

        if num_backbone_tokens is not None:
            hidden_backbone = hidden[:, :num_backbone_tokens, :]
            norm2_backbone = norm2_x[:, :num_backbone_tokens, :]
            if shared_lora is not None:
                hidden_backbone = hidden_backbone + shared_lora.fc1_lora(norm2_backbone)

            if x.shape[1] > num_backbone_tokens:
                hidden_expert = hidden[:, num_backbone_tokens:, :]
                if expert_lora is not None:
                    hidden_expert = hidden_expert + expert_lora.fc1_lora(
                        norm2_x[:, num_backbone_tokens:, :],
                        expert_fc1_proj,
                    )
                hidden = torch.cat((hidden_backbone, hidden_expert), dim=1)
            else:
                hidden = hidden_backbone

        hidden = self.mlp_drop(self.act(hidden))
        out = self.fc2(hidden)

        if num_backbone_tokens is not None:
            out_backbone = out[:, :num_backbone_tokens, :]
            if shared_lora is not None:
                out_backbone = out_backbone + shared_lora.fc2_lora(hidden[:, :num_backbone_tokens, :])

            if x.shape[1] > num_backbone_tokens:
                out_expert = out[:, num_backbone_tokens:, :]
                if expert_lora is not None:
                    out_expert = out_expert + expert_lora.fc2_lora(
                        hidden[:, num_backbone_tokens:, :],
                        expert_fc2_proj,
                    )
                out = torch.cat((out_backbone, out_expert), dim=1)
            else:
                out = out_backbone

        out = self.drop_path(self.mlp_drop(out))
        return residual + out


class VisionTransformer(TunaMaxVisionTransformer):
    """SPiE v13 ViT with shared LoRA and SVD-initialized expert VeRA bases."""

    def __init__(
        self,
        *args,
        expert_tokens=4,
        lora_rank=8,
        lora_alpha=1.0,
        shared_lora_rank=None,
        shared_lora_alpha=None,
        vera_rank=256,
        vera_dropout=0.0,
        vera_d_initial=0.1,
        vera_projection_seed=0,
        vera_save_projection=True,
        cassle_predictor_hidden_dim=None,
        **kwargs,
    ):
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.shared_lora_rank = int(shared_lora_rank if shared_lora_rank is not None else lora_rank)
        self.shared_lora_alpha = float(shared_lora_alpha if shared_lora_alpha is not None else lora_alpha)
        self.vera_rank = int(vera_rank)
        self.vera_dropout = float(vera_dropout)
        self.vera_d_initial = float(vera_d_initial)
        self.vera_projection_seed = int(vera_projection_seed)
        self.vera_save_projection = bool(vera_save_projection)
        self.cassle_predictor_hidden_dim = cassle_predictor_hidden_dim
        super().__init__(*args, **kwargs)
        if expert_tokens <= 0:
            raise ValueError(f"expert_tokens must be > 0, got {expert_tokens}")

        self.expert_tokens = int(expert_tokens)
        self.cur_expert_tokens = self._init_expert_tokens_from_cls()
        self.expert_token_list = nn.ParameterList()
        self.blocks = nn.ModuleList([MLPMixedAdapterBlock(block) for block in self.blocks])
        self.refresh_expert_vera_projections_from_backbone_weights()
        self.init_shared_adapters()
        self.cassle_predictor = CaSSLePredictor(self.embed_dim, hidden_dim=self.cassle_predictor_hidden_dim)
        self._mask_cache = {}

    def refresh_expert_vera_projections_from_backbone_weights(self):
        fc1_projections = nn.ModuleList()
        fc2_projections = nn.ModuleList()
        for block in self.blocks:
            fc1_projections.append(
                FrozenVeraProjection.from_linear_weight(
                    block.fc1.weight,
                    rank=self.vera_rank,
                    save_projection=self.vera_save_projection,
                )
            )
            fc2_projections.append(
                FrozenVeraProjection.from_linear_weight(
                    block.fc2.weight,
                    rank=self.vera_rank,
                    save_projection=self.vera_save_projection,
                )
            )
        self.expert_fc1_vera_projection = fc1_projections
        self.expert_fc2_vera_projection = fc2_projections

    def init_adapters(self):
        self.cur_adapter = nn.ModuleList()
        for block in self.blocks:
            adapter = MLPVeRAAdapter(
                dim=block.norm1.normalized_shape[0],
                mlp_hidden_dim=block.fc1.out_features,
                rank=self.vera_rank,
                dropout=self.vera_dropout,
                d_initial=self.vera_d_initial,
            ).to(self._device)
            self.cur_adapter.append(adapter)
        self.cur_adapter.requires_grad_(True)

    def init_shared_adapters(self):
        self.cur_shared_adapter = nn.ModuleList()
        for block in self.blocks:
            adapter = MLPLoRAAdapter(
                dim=block.norm1.normalized_shape[0],
                mlp_hidden_dim=block.fc1.out_features,
                rank=self.shared_lora_rank,
                alpha=self.shared_lora_alpha,
            ).to(self._device)
            self.cur_shared_adapter.append(adapter)
        self.cur_shared_adapter.requires_grad_(True)

    def _init_expert_tokens_from_cls(self):
        return nn.Parameter(self.cls_token.detach().clone().expand(1, self.expert_tokens, -1).clone())

    def reset_task_modules(self):
        self.init_adapters()
        self.cur_expert_tokens = self._init_expert_tokens_from_cls()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for adapter in self.cur_adapter:
            adapter.requires_grad_(True)
        self.cur_expert_tokens.requires_grad = True
        self.cur_shared_adapter.requires_grad_(True)
        self.cassle_predictor.requires_grad_(True)

    def adapter_update(self):
        frozen_adapter = copy.deepcopy(self.cur_adapter)
        frozen_adapter.requires_grad_(False)
        self.adapter_list.append(frozen_adapter)

        frozen_tokens = nn.Parameter(self.cur_expert_tokens.detach().clone(), requires_grad=False)
        self.expert_token_list.append(frozen_tokens)

    def _select_expert_tokens(self, adapter_id, train):
        if adapter_id == -1:
            return None
        if train or adapter_id == len(self.expert_token_list):
            return self.cur_expert_tokens
        if 0 <= adapter_id < len(self.expert_token_list):
            return self.expert_token_list[adapter_id]
        if adapter_id > len(self.expert_token_list):
            return self.cur_expert_tokens
        raise ValueError(f"Invalid adapter_id: {adapter_id}")

    def _select_expert_adapter_stack(self, adapter_id, train):
        if adapter_id == -1:
            return None
        if train or adapter_id == len(self.adapter_list):
            return self.cur_adapter
        if 0 <= adapter_id < len(self.adapter_list):
            return self.adapter_list[adapter_id]
        if adapter_id > len(self.adapter_list):
            return self.merged_adapter
        raise ValueError(f"Invalid adapter_id: {adapter_id}")

    def _build_attn_mask(self, num_backbone_tokens, num_expert_tokens, device, dtype):
        if num_expert_tokens <= 0:
            return None

        key = (num_backbone_tokens, num_expert_tokens, str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        total_tokens = num_backbone_tokens + num_expert_tokens
        mask = torch.full((total_tokens, total_tokens), float("-inf"), device=device, dtype=dtype)
        mask[:num_backbone_tokens, :num_backbone_tokens] = 0.0
        mask[num_backbone_tokens:, :num_backbone_tokens] = 0.0
        mask[num_backbone_tokens:, num_backbone_tokens:] = 0.0

        self._mask_cache[key] = mask
        return mask

    def forward_features(self, x, adapter_id, train):
        bsz = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        num_backbone_tokens = x.shape[1]

        expert_tokens = self._select_expert_tokens(adapter_id, train)
        if expert_tokens is not None:
            x = torch.cat((x, expert_tokens.expand(bsz, -1, -1)), dim=1)

        num_expert_tokens = 0 if expert_tokens is None else self.expert_tokens
        x = self.pos_drop(x)
        attn_mask = self._build_attn_mask(
            num_backbone_tokens=num_backbone_tokens,
            num_expert_tokens=num_expert_tokens,
            device=x.device,
            dtype=x.dtype,
        )
        expert_adapter_stack = self._select_expert_adapter_stack(adapter_id, train)
        shared_adapter_stack = self.cur_shared_adapter

        for layer_idx, blk in enumerate(self.blocks):
            expert_lora = None if expert_adapter_stack is None else expert_adapter_stack[layer_idx]
            x = blk(
                x,
                shared_lora=shared_adapter_stack[layer_idx],
                expert_lora=expert_lora,
                attn_mask=attn_mask,
                num_backbone_tokens=num_backbone_tokens,
                expert_fc1_proj=self.expert_fc1_vera_projection[layer_idx],
                expert_fc2_proj=self.expert_fc2_vera_projection[layer_idx],
            )

        if self.global_pool:
            x = x[:, 1:num_backbone_tokens, :].mean(dim=1)
            cls_features = self.fc_norm(x)
            expert_features = cls_features
        else:
            x = self.norm(x)
            cls_features = x[:, 0]
            if expert_tokens is None:
                expert_features = cls_features
            else:
                expert_features = x[:, -self.expert_tokens :, :].mean(dim=1)

        return {
            "x": expert_features,
            "pre_logits": expert_features,
            "features": expert_features,
            "cls_features": cls_features,
            "expert_features": expert_features,
        }

    def forward_head(self, res):
        x = res["x"]
        res["logits"] = self.head(x)
        return res

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        if fc_only:
            return {"logits": self.head(x)}

        res = self.forward_features(x, adapter_id, train)
        return self.forward_head(res)


def _load_pretrained_and_refresh_expert_svd(model, timm_name, num_classes):
    model = _load_pretrained_from_timm(model, timm_name, num_classes)
    model.refresh_expert_vera_projections_from_backbone_weights()
    return model


def _load_pretrained_from_timm(model, timm_name, num_classes):
    checkpoint_model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict.keys()):
        if "qkv.weight" in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768 : 768 * 2]
            v_weight = qkv_weight[768 * 2 :]
            state_dict[key.replace("qkv.weight", "q_proj.weight")] = q_weight
            state_dict[key.replace("qkv.weight", "k_proj.weight")] = k_weight
            state_dict[key.replace("qkv.weight", "v_proj.weight")] = v_weight
        elif "qkv.bias" in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768 : 768 * 2]
            v_bias = qkv_bias[768 * 2 :]
            state_dict[key.replace("qkv.bias", "q_proj.bias")] = q_bias
            state_dict[key.replace("qkv.bias", "k_proj.bias")] = k_bias
            state_dict[key.replace("qkv.bias", "v_proj.bias")] = v_bias
    for key in list(state_dict.keys()):
        if "mlp.fc" in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace("mlp.", "")] = fc_weight

    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        if "head" in name or "adapter" in name or "expert_token" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def vit_base_patch16_224_spie_v13(pretrained=False, **kwargs):
    del pretrained
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return _load_pretrained_and_refresh_expert_svd(model, "vit_base_patch16_224", kwargs["num_classes"])


def vit_base_patch16_224_in21k_spie_v13(pretrained=False, **kwargs):
    del pretrained
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return _load_pretrained_and_refresh_expert_svd(model, "vit_base_patch16_224_in21k", kwargs["num_classes"])


vit_base_patch16_224_spiev13 = vit_base_patch16_224_spie_v13
vit_base_patch16_224_in21k_spiev13 = vit_base_patch16_224_in21k_spie_v13
