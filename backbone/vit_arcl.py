from functools import partial

from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()

import timm
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        q = self._shape(self.q_proj(x), seq_len, batch_size)
        k = self._shape(self.k_proj(x), seq_len, batch_size)
        v = self._shape(self.v_proj(x), seq_len, batch_size)
        self.input = x.detach().to(q.dtype)

        self.q = q.detach()
        self.k = k.detach()
        self.v = v.detach()

        q_scaled = q * self.scale
        self.q_scaled = q_scaled.detach()

        attn_scores = q_scaled @ k.transpose(-2, -1)
        if self.training:
            self.attn_no_softmax = attn_scores
            self.attn_no_softmax.retain_grad()

        attn = attn_scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.training:
            self.attn = attn
            self.attn.retain_grad()
            self.attn_clone = attn.detach().clone()

        out = attn @ v
        if self.training:
            self.out = out
            self.out.retain_grad()

        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


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
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.mlp_drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.mlp_drop(self.fc2(x))
        x = residual + self.drop_path(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.out_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
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
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity() if num_classes == 0 else nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        feats = self.forward_features(x)
        return {"features": feats, "logits": self.head(feats)}


def _load_pretrained(backbone_name, pretrained=True, **kwargs):
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
    if pretrained:
        checkpoint_model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
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

        model.load_state_dict(state_dict, strict=False)
    for name, param in model.named_parameters():
        param.requires_grad = any(token in name for token in ("q_proj.weight", "k_proj.weight", "v_proj.weight"))
    return model.eval()


def vit_base_patch16_224_arcl(pretrained=False, **kwargs):
    return _load_pretrained("vit_base_patch16_224", pretrained=pretrained, **kwargs)


def vit_base_patch16_224_in21k_arcl(pretrained=False, **kwargs):
    return _load_pretrained("vit_base_patch16_224_in21k", pretrained=pretrained, **kwargs)


def pretrained_vit_b16_224_arcl(pretrained=False, **kwargs):
    return vit_base_patch16_224_arcl(pretrained=pretrained, **kwargs)


def pretrained_vit_b16_224_in21k_arcl(pretrained=False, **kwargs):
    return vit_base_patch16_224_in21k_arcl(pretrained=pretrained, **kwargs)
