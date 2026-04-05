import copy
from functools import partial

import timm
import torch
import torch.nn as nn

from backbone.vit_tunamax import VisionTransformer as TunaMaxVisionTransformer


class ExpertIsolatedBlock(nn.Module):
    """SPiE-style block wrapper with attention masking and expert-only adapters."""

    def __init__(self, base_block, expert_tokens):
        super().__init__()
        self.config = base_block.config
        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self.drop_path = base_block.drop_path
        self.norm2 = base_block.norm2
        self.fc1 = base_block.fc1
        self.fc2 = base_block.fc2
        self.act = base_block.act
        self.mlp_drop = base_block.mlp_drop
        self.expert_tokens = int(expert_tokens)

    def _forward_attention(self, x, attn_mask=None):
        if attn_mask is None:
            return self.attn(x)

        bsz, seq_len, dim = x.shape
        num_heads = self.attn.num_heads
        head_dim = self.attn.head_dim

        q = self.attn.q_proj(x)
        q = self.attn._shape(q, seq_len, bsz).view(bsz * num_heads, seq_len, head_dim)
        k = self.attn._shape(self.attn.k_proj(x), -1, bsz).view(bsz * num_heads, -1, head_dim)
        v = self.attn._shape(self.attn.v_proj(x), -1, bsz).view(bsz * num_heads, -1, head_dim)

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

    def forward(self, x, adapt=None, attn_mask=None, num_backbone_tokens=None):
        x = x + self.drop_path(self._forward_attention(self.norm1(x), attn_mask=attn_mask))

        adapt_x = None
        if adapt is not None and num_backbone_tokens is not None and x.shape[1] > num_backbone_tokens:
            expert_x = x[:, num_backbone_tokens:, :]
            adapt_x = adapt(expert_x, add_residual=False)

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if adapt_x is not None and self.config.ffn_adapt:
            if self.config.ffn_option == "sequential":
                x[:, num_backbone_tokens:, :] = adapt(x[:, num_backbone_tokens:, :])
            elif self.config.ffn_option == "parallel":
                x[:, num_backbone_tokens:, :] = x[:, num_backbone_tokens:, :] + adapt_x
            else:
                raise ValueError(self.config.ffn_option)

        x = residual + x
        return x


class VisionTransformer(TunaMaxVisionTransformer):
    """TunaMax-style ViT with per-adapter expert tokens."""

    def __init__(self, *args, expert_tokens=4, **kwargs):
        super().__init__(*args, **kwargs)
        if expert_tokens <= 0:
            raise ValueError(f"expert_tokens must be > 0, got {expert_tokens}")

        self.expert_tokens = int(expert_tokens)
        self.cur_expert_tokens = self._init_expert_tokens_from_cls()
        self.expert_token_list = nn.ParameterList()
        self.blocks = nn.ModuleList([ExpertIsolatedBlock(block, self.expert_tokens) for block in self.blocks])
        self._mask_cache = {}

    def _init_expert_tokens_from_cls(self):
        return nn.Parameter(self.cls_token.detach().clone().expand(1, self.expert_tokens, -1).clone())

    def reset_task_modules(self):
        # Each incremental task should get a fresh expert branch instead of
        # continuing to optimize the previous task's expert parameters.
        self.init_adapters()
        self.cur_expert_tokens = self._init_expert_tokens_from_cls()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for adapter in self.cur_adapter:
            adapter.requires_grad_(True)
        self.cur_expert_tokens.requires_grad = True

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

    def _select_adapter_stack(self, adapter_id, train):
        if adapter_id == -1:
            return None
        if train:
            return self.cur_adapter
        if adapter_id == len(self.adapter_list):
            return self.cur_adapter
        if 0 <= adapter_id < len(self.adapter_list):
            return self.adapter_list[adapter_id]
        if adapter_id > len(self.adapter_list):
            return self.merged_adapter
        raise ValueError(f"Invalid adapter_id: {adapter_id}")

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

        x = self.pos_drop(x)
        attn_mask = self._build_attn_mask(
            num_backbone_tokens=num_backbone_tokens,
            num_expert_tokens=0 if expert_tokens is None else self.expert_tokens,
            device=x.device,
            dtype=x.dtype,
        )
        adapter_stack = self._select_adapter_stack(adapter_id, train)

        for layer_idx, blk in enumerate(self.blocks):
            adapt = None if adapter_stack is None else adapter_stack[layer_idx]
            x = blk(
                x,
                adapt=adapt,
                attn_mask=attn_mask,
                num_backbone_tokens=num_backbone_tokens,
            )

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            cls_features = self.fc_norm(x)
            expert_features = cls_features
        else:
            x = self.norm(x)
            cls_features = x[:, 0]
            if expert_tokens is None:
                expert_features = cls_features
            else:
                expert_features = x[:, -self.expert_tokens :, :].mean(dim=1)

        res = {
            "x": expert_features,
            "pre_logits": expert_features,
            "features": expert_features,
            "cls_features": cls_features,
            "expert_features": expert_features,
        }
        return res


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


def vit_base_patch16_224_spie_v2(pretrained=False, **kwargs):
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


def vit_base_patch16_224_in21k_spie_v2(pretrained=False, **kwargs):
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
