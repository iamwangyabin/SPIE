from functools import partial

import torch
import torch.nn as nn

from backbone.vit_spie_v11 import MLPVeRAAdapter
from backbone.vit_spie_v11 import VisionTransformer as SPiEV11VisionTransformer
from backbone.vit_spie_v11 import _load_pretrained_from_timm


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


class VisionTransformer(SPiEV11VisionTransformer):
    """SPiE v13 ViT with shared LoRA and SVD-initialized expert VeRA bases."""

    def _init_expert_vera_projections(self):
        self.refresh_expert_vera_projections_from_backbone_weights()

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


def _load_pretrained_and_refresh_expert_svd(model, timm_name, num_classes):
    model = _load_pretrained_from_timm(model, timm_name, num_classes)
    model.refresh_expert_vera_projections_from_backbone_weights()
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
