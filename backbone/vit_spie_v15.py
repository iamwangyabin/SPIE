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
    """SPiE v15 ViT with shared LoRA and parallel multi-expert inference."""

    def __init__(
        self,
        *args,
        expert_tokens=4,
        expert_residual_scale=0.5,
        shared_lora_rank=8,
        shared_lora_alpha=1.0,
        vera_rank=256,
        vera_dropout=0.0,
        vera_d_initial=0.1,
        vera_save_projection=True,
        **kwargs,
    ):
        self.shared_lora_rank = int(shared_lora_rank)
        self.shared_lora_alpha = float(shared_lora_alpha)
        self.vera_rank = int(vera_rank)
        self.vera_dropout = float(vera_dropout)
        self.vera_d_initial = float(vera_d_initial)
        self.vera_save_projection = bool(vera_save_projection)
        super().__init__(*args, **kwargs)
        if expert_tokens <= 0:
            raise ValueError(f"expert_tokens must be > 0, got {expert_tokens}")

        self.expert_tokens = int(expert_tokens)
        self.expert_residual_scale = float(expert_residual_scale)
        self.cur_expert_tokens = self._init_expert_tokens_from_cls()
        self.expert_token_list = nn.ParameterList()
        self.blocks = nn.ModuleList([MLPMixedAdapterBlock(block) for block in self.blocks])
        self.refresh_expert_vera_projections_from_backbone_weights()
        self.init_shared_adapters()
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

    def _fuse_shared_expert_features(self, cls_features, expert_tokens):
        if expert_tokens is None:
            return cls_features

        if expert_tokens.ndim == 3:
            expert_summary = expert_tokens[:, 0, :]
        elif expert_tokens.ndim == 4:
            expert_summary = expert_tokens[:, :, 0, :]
        else:
            raise ValueError(f"Unsupported expert token shape: {tuple(expert_tokens.shape)}")

        return torch.cat((cls_features, expert_summary), dim=-1)

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
                expert_token_features = x[:, -self.expert_tokens :, :]
                expert_features = self._fuse_shared_expert_features(cls_features, expert_token_features)

        return {
            "x": cls_features,
            "pre_logits": cls_features,
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

    def _stack_expert_tokens(self, adapter_ids, train=False):
        tokens = []
        for adapter_id in adapter_ids:
            expert_tokens = self._select_expert_tokens(adapter_id, train=train)
            if expert_tokens is None:
                raise ValueError("Parallel expert inference requires valid expert token ids.")
            tokens.append(expert_tokens)
        return torch.stack(tokens, dim=0)

    def _stack_expert_adapters(self, adapter_ids, layer_idx, train=False):
        adapters = []
        for adapter_id in adapter_ids:
            adapter_stack = self._select_expert_adapter_stack(adapter_id, train=train)
            if adapter_stack is None:
                raise ValueError("Parallel expert inference requires valid expert adapter ids.")
            adapters.append(adapter_stack[layer_idx])
        return adapters

    def _forward_backbone_attention(self, blk, backbone_tokens):
        return backbone_tokens + blk.drop_path(blk._forward_attention(blk.norm1(backbone_tokens)))

    def _forward_backbone_mlp(self, blk, backbone_tokens, shared_lora):
        residual = backbone_tokens
        norm2_x = blk.norm2(backbone_tokens)
        hidden = blk.fc1(norm2_x)
        hidden = hidden + shared_lora.fc1_lora(norm2_x)
        hidden = blk.mlp_drop(blk.act(hidden))

        out = blk.fc2(hidden)
        out = out + shared_lora.fc2_lora(hidden)
        out = blk.drop_path(blk.mlp_drop(out))
        return residual + out

    def _forward_expert_attention_parallel(self, blk, backbone_tokens, expert_tokens):
        expert_count, batch_size, num_expert_tokens, dim = expert_tokens.shape
        num_backbone_tokens = backbone_tokens.shape[1]
        num_heads = blk.attn.num_heads
        head_dim = blk.attn.head_dim

        norm_backbone = blk.norm1(backbone_tokens)
        norm_backbone = norm_backbone.unsqueeze(0).expand(expert_count, -1, -1, -1)

        expert_flat = expert_tokens.reshape(expert_count * batch_size, num_expert_tokens, dim)
        norm_expert = blk.norm1(expert_flat).reshape(expert_count, batch_size, num_expert_tokens, dim)

        q = blk.attn.q_proj(norm_expert.reshape(expert_count * batch_size, num_expert_tokens, dim))
        q = blk.attn._shape(q, num_expert_tokens, expert_count * batch_size).view(
            expert_count * batch_size * num_heads,
            num_expert_tokens,
            head_dim,
        )

        kv_input = torch.cat((norm_backbone, norm_expert), dim=2)
        kv_flat = kv_input.reshape(expert_count * batch_size, num_backbone_tokens + num_expert_tokens, dim)

        k = blk.attn.k_proj(kv_flat)
        v = blk.attn.v_proj(kv_flat)
        k = blk.attn._shape(k, num_backbone_tokens + num_expert_tokens, expert_count * batch_size).view(
            expert_count * batch_size * num_heads,
            num_backbone_tokens + num_expert_tokens,
            head_dim,
        )
        v = blk.attn._shape(v, num_backbone_tokens + num_expert_tokens, expert_count * batch_size).view(
            expert_count * batch_size * num_heads,
            num_backbone_tokens + num_expert_tokens,
            head_dim,
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * blk.attn.scale
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = blk.attn.attn_drop(attn_probs)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(expert_count * batch_size, num_heads, num_expert_tokens, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(expert_count * batch_size, num_expert_tokens, dim)
        attn_output = blk.attn.proj(attn_output)
        attn_output = blk.attn.proj_drop(attn_output)
        return attn_output.reshape(expert_count, batch_size, num_expert_tokens, dim)

    def _apply_batched_vera(self, x, expert_loras, shared_proj):
        if not expert_loras:
            return torch.zeros_like(x)

        expert_count = len(expert_loras)
        dtype = expert_loras[0].lambda_d.dtype
        device = x.device

        x_adapt = x.to(dtype=dtype)
        dropout_p = float(expert_loras[0].dropout.p)
        if dropout_p > 0:
            x_adapt = F.dropout(x_adapt, p=dropout_p, training=self.training)

        a = shared_proj.A.to(device=device, dtype=dtype)
        b = shared_proj.B.to(device=device, dtype=dtype)
        lambda_d = torch.stack([expert_lora.lambda_d for expert_lora in expert_loras], dim=0).to(
            device=device, dtype=dtype
        )
        lambda_b = torch.stack([expert_lora.lambda_b for expert_lora in expert_loras], dim=0).to(
            device=device, dtype=dtype
        )

        if lambda_d.shape[0] != expert_count:
            raise ValueError("Batched VeRA parameter stack shape mismatch.")

        z = torch.einsum("ebtd,rd->ebtr", x_adapt, a)
        z = z * lambda_d[:, None, None, :]
        delta = torch.einsum("ebtr,or->ebto", z, b)
        delta = delta * lambda_b[:, None, None, :]
        return delta.to(dtype=x.dtype)

    def _forward_expert_mlp_parallel(
        self,
        blk,
        expert_tokens,
        expert_adapters,
        expert_fc1_proj,
        expert_fc2_proj,
    ):
        residual = expert_tokens
        expert_count, batch_size, num_expert_tokens, dim = expert_tokens.shape

        norm2_x = blk.norm2(expert_tokens.reshape(expert_count * batch_size, num_expert_tokens, dim))
        norm2_x = norm2_x.reshape(expert_count, batch_size, num_expert_tokens, dim)

        hidden = blk.fc1(norm2_x.reshape(expert_count * batch_size, num_expert_tokens, dim))
        hidden = hidden.reshape(expert_count, batch_size, num_expert_tokens, -1)
        hidden = hidden + self._apply_batched_vera(
            norm2_x,
            [adapter.fc1_lora for adapter in expert_adapters],
            expert_fc1_proj,
        )
        hidden = blk.act(hidden)
        hidden = blk.mlp_drop(hidden.reshape(expert_count * batch_size, num_expert_tokens, -1))
        hidden = hidden.reshape(expert_count, batch_size, num_expert_tokens, -1)

        out = blk.fc2(hidden.reshape(expert_count * batch_size, num_expert_tokens, -1))
        out = out.reshape(expert_count, batch_size, num_expert_tokens, dim)
        out = out + self._apply_batched_vera(
            hidden,
            [adapter.fc2_lora for adapter in expert_adapters],
            expert_fc2_proj,
        )
        out = blk.mlp_drop(out.reshape(expert_count * batch_size, num_expert_tokens, dim))
        out = blk.drop_path(out).reshape(expert_count, batch_size, num_expert_tokens, dim)
        return residual + out

    @torch.no_grad()
    def forward_multi_expert_features(self, x, adapter_ids):
        if self.global_pool:
            raise NotImplementedError("Parallel expert inference currently expects global_pool=False.")
        if not adapter_ids:
            raise ValueError("adapter_ids must be non-empty for parallel expert inference.")

        batch_size = x.shape[0]
        backbone_tokens = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        backbone_tokens = torch.cat((cls_tokens, backbone_tokens), dim=1)
        backbone_tokens = backbone_tokens + self.pos_embed
        backbone_tokens = self.pos_drop(backbone_tokens)

        expert_tokens = self._stack_expert_tokens(adapter_ids, train=False)
        expert_tokens = expert_tokens.expand(-1, batch_size, -1, -1).contiguous()
        expert_tokens = self.pos_drop(expert_tokens)

        for layer_idx, blk in enumerate(self.blocks):
            backbone_attended = self._forward_backbone_attention(blk, backbone_tokens)
            expert_adapters = self._stack_expert_adapters(adapter_ids, layer_idx, train=False)
            expert_attended = expert_tokens + blk.drop_path(
                self._forward_expert_attention_parallel(blk, backbone_tokens, expert_tokens)
            )
            backbone_tokens = self._forward_backbone_mlp(blk, backbone_attended, self.cur_shared_adapter[layer_idx])
            expert_tokens = self._forward_expert_mlp_parallel(
                blk,
                expert_attended,
                expert_adapters,
                self.expert_fc1_vera_projection[layer_idx],
                self.expert_fc2_vera_projection[layer_idx],
            )

        backbone_tokens = self.norm(backbone_tokens)
        expert_count, _, num_expert_tokens, dim = expert_tokens.shape
        expert_tokens = self.norm(expert_tokens.reshape(expert_count * batch_size, num_expert_tokens, dim))
        expert_tokens = expert_tokens.reshape(expert_count, batch_size, num_expert_tokens, dim)

        cls_features = backbone_tokens[:, 0]
        expert_features = self._fuse_shared_expert_features(cls_features.unsqueeze(0).expand(expert_count, -1, -1), expert_tokens)
        return {
            "cls_features": cls_features,
            "expert_features": expert_features,
            "features": expert_features,
        }


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


def vit_base_patch16_224_spie_v15(pretrained=False, **kwargs):
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


def vit_base_patch16_224_in21k_spie_v15(pretrained=False, **kwargs):
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
