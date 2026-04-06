from functools import partial

import torch
import torch.nn as nn

from backbone.vit_spie_v3 import VisionTransformer as SPiEV3VisionTransformer
from backbone.vit_spie_v3 import _load_pretrained_from_timm


class VisionTransformer(SPiEV3VisionTransformer):
    """SPiE v4 backbone with CLS and pooled expert-token concatenation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim = self.embed_dim * 2

    def forward_features(self, x, adapter_id, train):
        bsz = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        num_backbone_tokens = x.shape[1]

        expert_tokens = self._select_expert_tokens(adapter_id, train)
        num_expert_tokens = 0 if expert_tokens is None else self.expert_tokens
        if expert_tokens is not None:
            x = torch.cat((expert_tokens.expand(bsz, -1, -1), x), dim=1)

        x = self.pos_drop(x)
        attn_mask = self._build_attn_mask(
            num_backbone_tokens=num_backbone_tokens,
            num_expert_tokens=num_expert_tokens,
            device=x.device,
            dtype=x.dtype,
        )
        adapter_stack = self._select_adapter_stack(adapter_id, train)

        for layer_idx, blk in enumerate(self.blocks):
            lora = None if adapter_stack is None else adapter_stack[layer_idx]
            x = blk(
                x,
                lora=lora,
                attn_mask=attn_mask,
                num_expert_tokens=num_expert_tokens,
            )

        if self.global_pool:
            x = x[:, num_expert_tokens + 1 :, :].mean(dim=1)
            cls_features = self.fc_norm(x)
            expert_features = cls_features
        else:
            x = self.norm(x)
            cls_features = x[:, num_expert_tokens]
            if expert_tokens is None:
                expert_features = cls_features
            else:
                expert_features = x[:, :num_expert_tokens, :].mean(dim=1)

        if expert_tokens is None:
            fused_features = torch.cat((cls_features, cls_features), dim=1)
        else:
            fused_features = torch.cat((cls_features, expert_features), dim=1)

        return {
            "x": fused_features,
            "pre_logits": fused_features,
            "features": fused_features,
            "cls_features": cls_features,
            "expert_features": expert_features,
            "fused_features": fused_features,
        }

    def forward_head(self, res):
        res["logits"] = res["x"]
        return res

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        if fc_only:
            return {"logits": x}

        res = self.forward_features(x, adapter_id, train)
        res = self.forward_head(res)
        return res


def vit_base_patch16_224_spie_v4(pretrained=False, **kwargs):
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


def vit_base_patch16_224_in21k_spie_v4(pretrained=False, **kwargs):
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
