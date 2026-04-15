import copy
from functools import partial

import torch
import torch.nn as nn
from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()
import timm

from backbone.vit_tuna import VisionTransformer as TunaVisionTransformer


class VisionTransformer(TunaVisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_ema = None

    def adapter_merge(self):
        momentum = float(getattr(self.config, "adapter_momentum", 0.0) or 0.0)
        if momentum <= 0:
            return

        if self.adapter_ema is None:
            self.adapter_ema = copy.deepcopy(self.cur_adapter).to(self._device)
            return

        with torch.no_grad():
            for ema_layer, cur_layer in zip(self.adapter_ema, self.cur_adapter):
                for ema_param, cur_param in zip(ema_layer.parameters(), cur_layer.parameters()):
                    ema_param.data.mul_(momentum).add_(cur_param.data, alpha=1.0 - momentum)
                    cur_param.data.copy_(ema_param.data)


def _load_pretrained_weights(model, model_name, num_classes):
    checkpoint_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    state_dict = checkpoint_model.state_dict()

    for key in list(state_dict.keys()):
        if "qkv.weight" in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace("qkv.weight", "q_proj.weight")] = qkv_weight[:768]
            state_dict[key.replace("qkv.weight", "k_proj.weight")] = qkv_weight[768:768 * 2]
            state_dict[key.replace("qkv.weight", "v_proj.weight")] = qkv_weight[768 * 2:]
        elif "qkv.bias" in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace("qkv.bias", "q_proj.bias")] = qkv_bias[:768]
            state_dict[key.replace("qkv.bias", "k_proj.bias")] = qkv_bias[768:768 * 2]
            state_dict[key.replace("qkv.bias", "v_proj.bias")] = qkv_bias[768 * 2:]

    for key in list(state_dict.keys()):
        if "mlp.fc" in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace("mlp.", "")] = fc_weight

    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = "head" in name or "adapter" in name

    return model


def vit_base_patch16_224_mos(pretrained=False, **kwargs):
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
    return _load_pretrained_weights(model, "vit_base_patch16_224", kwargs["num_classes"])


def vit_base_patch16_224_in21k_mos(pretrained=False, **kwargs):
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
    return _load_pretrained_weights(model, "vit_base_patch16_224_in21k", kwargs["num_classes"])
