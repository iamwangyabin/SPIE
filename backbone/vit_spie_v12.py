import copy
from functools import partial

import torch
import torch.nn as nn

from backbone.vit_spie_v11 import MLPVeRAAdapter
from backbone.vit_spie_v11 import SharedVeraProjection
from backbone.vit_spie_v11 import _load_pretrained_from_timm
from backbone.vit_spie_v6 import CaSSLePredictor
from backbone.vit_tunamax import VisionTransformer as TunaMaxVisionTransformer


class MLPDualVeRAAdapterBlock(nn.Module):
    """TunaMax block wrapper with shared-VeRA and expert-VeRA MLP branches."""

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
        shared_fc1_proj=None,
        shared_fc2_proj=None,
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
                hidden_backbone = hidden_backbone + shared_lora.fc1_lora(norm2_backbone, shared_fc1_proj)

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
                out_backbone = out_backbone + shared_lora.fc2_lora(hidden[:, :num_backbone_tokens, :], shared_fc2_proj)

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
    """ViT with shared/expert VeRA MLP branches plus isolated expert tokens."""

    def __init__(
        self,
        *args,
        expert_tokens=4,
        vera_rank=256,
        vera_dropout=0.0,
        vera_d_initial=0.1,
        vera_projection_seed=0,
        vera_save_projection=True,
        shared_vera_rank=None,
        shared_vera_dropout=None,
        shared_vera_d_initial=None,
        shared_vera_projection_seed=None,
        shared_vera_save_projection=None,
        cassle_predictor_hidden_dim=None,
        **kwargs,
    ):
        self.vera_rank = int(vera_rank)
        self.vera_dropout = float(vera_dropout)
        self.vera_d_initial = float(vera_d_initial)
        self.vera_projection_seed = int(vera_projection_seed)
        self.vera_save_projection = bool(vera_save_projection)

        self.shared_vera_rank = int(shared_vera_rank if shared_vera_rank is not None else vera_rank)
        self.shared_vera_dropout = float(shared_vera_dropout if shared_vera_dropout is not None else vera_dropout)
        self.shared_vera_d_initial = float(
            shared_vera_d_initial if shared_vera_d_initial is not None else vera_d_initial
        )
        self.shared_vera_projection_seed = int(
            shared_vera_projection_seed if shared_vera_projection_seed is not None else vera_projection_seed
        )
        self.shared_vera_save_projection = bool(
            shared_vera_save_projection if shared_vera_save_projection is not None else vera_save_projection
        )
        self.cassle_predictor_hidden_dim = cassle_predictor_hidden_dim
        super().__init__(*args, **kwargs)
        if expert_tokens <= 0:
            raise ValueError(f"expert_tokens must be > 0, got {expert_tokens}")

        self.expert_tokens = int(expert_tokens)
        self.cur_expert_tokens = self._init_expert_tokens_from_cls()
        self.expert_token_list = nn.ParameterList()
        self.blocks = nn.ModuleList([MLPDualVeRAAdapterBlock(block) for block in self.blocks])
        self._init_shared_vera_projections()
        self._init_expert_vera_projections()
        self.init_shared_adapters()
        self.cassle_predictor = CaSSLePredictor(self.embed_dim, hidden_dim=self.cassle_predictor_hidden_dim)
        self._mask_cache = {}

    def _init_shared_vera_projections(self):
        first_block = self.blocks[0]
        dim = first_block.norm1.normalized_shape[0]
        mlp_hidden_dim = first_block.fc1.out_features
        self.shared_fc1_vera_projection = SharedVeraProjection(
            max_in_features=dim,
            max_out_features=mlp_hidden_dim,
            r=self.shared_vera_rank,
            seed=self.shared_vera_projection_seed,
            save_projection=self.shared_vera_save_projection,
        )
        self.shared_fc2_vera_projection = SharedVeraProjection(
            max_in_features=mlp_hidden_dim,
            max_out_features=dim,
            r=self.shared_vera_rank,
            seed=self.shared_vera_projection_seed,
            save_projection=self.shared_vera_save_projection,
        )

    def _init_expert_vera_projections(self):
        first_block = self.blocks[0]
        dim = first_block.norm1.normalized_shape[0]
        mlp_hidden_dim = first_block.fc1.out_features
        self.expert_fc1_vera_projection = SharedVeraProjection(
            max_in_features=dim,
            max_out_features=mlp_hidden_dim,
            r=self.vera_rank,
            seed=self.vera_projection_seed,
            save_projection=self.vera_save_projection,
        )
        self.expert_fc2_vera_projection = SharedVeraProjection(
            max_in_features=mlp_hidden_dim,
            max_out_features=dim,
            r=self.vera_rank,
            seed=self.vera_projection_seed,
            save_projection=self.vera_save_projection,
        )

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
            adapter = MLPVeRAAdapter(
                dim=block.norm1.normalized_shape[0],
                mlp_hidden_dim=block.fc1.out_features,
                rank=self.shared_vera_rank,
                dropout=self.shared_vera_dropout,
                d_initial=self.shared_vera_d_initial,
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
                shared_fc1_proj=self.shared_fc1_vera_projection,
                shared_fc2_proj=self.shared_fc2_vera_projection,
                expert_fc1_proj=self.expert_fc1_vera_projection,
                expert_fc2_proj=self.expert_fc2_vera_projection,
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


def vit_base_patch16_224_spie_v12(pretrained=False, **kwargs):
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
    return _load_pretrained_from_timm(model, "vit_base_patch16_224", kwargs["num_classes"])


def vit_base_patch16_224_in21k_spie_v12(pretrained=False, **kwargs):
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
    return _load_pretrained_from_timm(model, "vit_base_patch16_224_in21k", kwargs["num_classes"])


vit_base_patch16_224_spiev12 = vit_base_patch16_224_spie_v12
vit_base_patch16_224_in21k_spiev12 = vit_base_patch16_224_in21k_spie_v12
