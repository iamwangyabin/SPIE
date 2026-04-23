import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()
import timm
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, batch_size).view(batch_size * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, batch_size).view(batch_size * self.num_heads, -1, self.head_dim)
        q = self._shape(q, seq_len, batch_size).view(batch_size * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, channels)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Adapter(nn.Module):
    def __init__(
        self,
        config=None,
        d_model=None,
        bottleneck=None,
        dropout=0.0,
        init_option="bert",
        adapter_scalar="1.0",
        adapter_layernorm_option="in",
    ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option in {"in", "out"}:
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option == "bert":
            raise NotImplementedError
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == "in":
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down) * self.scale

        if self.adapter_layernorm_option == "out":
            up = self.adapter_layer_norm_before(up)

        return up + residual if add_residual else up


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        config=None,
        layer_id=None,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt and self.layer_id <= 12:
            self.adaptmlp = Adapter(
                self.config,
                dropout=0.1,
                bottleneck=config.ffn_num,
                init_option=config.ffn_adapter_init_option,
                adapter_scalar=config.ffn_adapter_scalar,
                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
            )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        if self.config.ffn_adapt and self.config.ffn_option == "parallel" and self.layer_id <= 12:
            adapt_x = self.adaptmlp(x, add_residual=False)

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt and self.layer_id <= 12:
            if self.config.ffn_option == "sequential":
                x = self.adaptmlp(x)
            elif self.config.ffn_option == "parallel":
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_option)

        return residual + x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        global_pool=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        tuning_config=None,
    ):
        super().__init__()
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    config=tuning_config,
                    layer_id=i,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0
            self.embeddings = nn.ParameterList(
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in range(depth)]
            )
            for embedding in self.embeddings:
                torch.nn.init.xavier_uniform_(embedding.data)

    def init_weights(self, mode=""):
        raise NotImplementedError()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, block in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                prompt = self.embeddings[idx].expand(batch_size, -1, -1)
                x = torch.cat([prompt, x], dim=1)
            x = block(x)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num :, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            return self.fc_norm(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
            return (x + x_dist) / 2
        return self.head(x)


def _load_pretrained_adapter(model, checkpoint_model):
    state_dict = checkpoint_model.state_dict()

    for key in list(state_dict.keys()):
        if "qkv.weight" in key:
            qkv_weight = state_dict.pop(key)
            state_dict[key.replace("qkv.weight", "q_proj.weight")] = qkv_weight[:768]
            state_dict[key.replace("qkv.weight", "k_proj.weight")] = qkv_weight[768 : 768 * 2]
            state_dict[key.replace("qkv.weight", "v_proj.weight")] = qkv_weight[768 * 2 :]
        elif "qkv.bias" in key:
            qkv_bias = state_dict.pop(key)
            state_dict[key.replace("qkv.bias", "q_proj.bias")] = qkv_bias[:768]
            state_dict[key.replace("qkv.bias", "k_proj.bias")] = qkv_bias[768 : 768 * 2]
            state_dict[key.replace("qkv.bias", "v_proj.bias")] = qkv_bias[768 * 2 :]

    for key in list(state_dict.keys()):
        if "mlp.fc" in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace("mlp.", "")] = fc_weight

    msg = model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        param.requires_grad = name in msg.missing_keys
    return model


def vit_base_patch16_224_adapter(pretrained=False, **kwargs):
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
    checkpoint_model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    return _load_pretrained_adapter(model, checkpoint_model)


def vit_base_patch16_224_in21k_adapter(pretrained=False, **kwargs):
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
    checkpoint_model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
    return _load_pretrained_adapter(model, checkpoint_model)
