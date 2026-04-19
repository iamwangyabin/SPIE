from collections import OrderedDict

import torch
from torch import Tensor, nn

from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()
import timm


class Prompt(nn.Module):
    def __init__(self, prompt_len: int, embed_dim: int, prompt_init: str = "uniform"):
        super().__init__()
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.prompt = nn.Parameter(torch.empty(1, prompt_len, embed_dim))
        self.reset_parameters(prompt_init)

    def reset_parameters(self, prompt_init: str):
        if prompt_init == "zero":
            nn.init.zeros_(self.prompt)
        elif prompt_init == "trunc_normal":
            nn.init.trunc_normal_(self.prompt, std=0.02)
        elif prompt_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.prompt)
        elif prompt_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.prompt, a=1.0)
        else:
            nn.init.uniform_(self.prompt, -1.0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x, self.prompt.expand(x.shape[0], -1, -1)], dim=1)


class VPTNSP2PPBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224_in21k",
        pretrained: bool = True,
        prompt_len: int = 4,
        prompt_start_block: int = 0,
        prompt_end_block: int = 11,
        prompt_init: str = "uniform",
    ):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = getattr(self.base_model, "num_features", 768)
        self.pretrained_cfg = getattr(self.base_model, "pretrained_cfg", {})
        self.prompt_len = prompt_len
        self.prompt_start_block = prompt_start_block
        self.prompt_end_block = prompt_end_block
        self.prompt_layers = nn.ModuleDict()

        if self.prompt_len > 0:
            for block_idx in range(prompt_start_block, prompt_end_block + 1):
                self.prompt_layers[str(block_idx)] = Prompt(prompt_len, self.out_dim, prompt_init=prompt_init)

    def prompt_items(self):
        for layer_name, prompt in self.prompt_layers.items():
            yield int(layer_name), prompt.prompt

    def prompt_param_id_dict(self):
        result = OrderedDict()
        for layer_idx, param in self.prompt_items():
            result[id(param)] = {
                "layer": layer_idx,
                "name": f"prompt_layers.{layer_idx}.prompt",
                "shape": list(param.shape),
            }
        return result

    def _accumulate_null_stats(self, block, prompt_param: Tensor, q: Tensor, attn: Tensor, src_tokens: int, store: dict):
        pid = id(prompt_param)
        if pid not in store:
            store[pid] = {}

        embed_dim = block.attn.qkv.in_features
        num_heads = block.attn.num_heads
        head_dim = embed_dim // num_heads
        w_qkv = block.attn.qkv.weight.detach()
        w_k = w_qkv[embed_dim : 2 * embed_dim].reshape(num_heads, head_dim, embed_dim)

        q_src = q[:, :, :src_tokens, :]
        q_src = torch.einsum("bhnd,hdf->bhnf", q_src, w_k)
        q_src = q_src.reshape(-1, embed_dim)
        cov_q = (q_src.T @ q_src) / max(q_src.shape[0], 1)

        attn_prompt = attn[:, :, :src_tokens, src_tokens:]
        attn_prompt = attn_prompt.reshape(-1, self.prompt_len)
        cov_prompt = (attn_prompt.T @ attn_prompt) / max(attn_prompt.shape[0], 1)

        if "interm_reader_1" not in store[pid]:
            store[pid]["interm_reader_1"] = torch.zeros_like(cov_q)
        if "interm_reader_2" not in store[pid]:
            store[pid]["interm_reader_2"] = torch.zeros_like(cov_prompt)
        store[pid]["interm_reader_1"] += cov_q
        store[pid]["interm_reader_2"] += cov_prompt

    def _forward_prompt_block(self, block, x: Tensor, prompt_layer: Prompt, record_null_stats: dict = None) -> Tensor:
        src_tokens = x.shape[1]
        x_prompt = prompt_layer(x)
        x_prompt = block.norm1(x_prompt)

        batch_size, token_num, embed_dim = x_prompt.shape
        num_heads = block.attn.num_heads
        head_dim = embed_dim // num_heads
        qkv = block.attn.qkv(x_prompt).reshape(batch_size, token_num, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = block.attn.q_norm(q)
        k = block.attn.k_norm(k)
        q = q * block.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        if record_null_stats is not None:
            self._accumulate_null_stats(block, prompt_layer.prompt, q / block.attn.scale, attn, src_tokens, record_null_stats)

        x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(batch_size, token_num, embed_dim)
        x_attn = block.attn.proj(x_attn)
        x_attn = block.attn.proj_drop(x_attn)
        x_attn = x_attn[:, :src_tokens]

        x = x + block.drop_path1(block.ls1(x_attn))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def forward(self, x: Tensor, train: bool = False, record_null_stats: dict = None):
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)
        x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)

        for block_idx, block in enumerate(self.base_model.blocks):
            prompt_layer = self.prompt_layers[str(block_idx)] if str(block_idx) in self.prompt_layers else None
            if prompt_layer is not None:
                x = self._forward_prompt_block(block, x, prompt_layer, record_null_stats=record_null_stats)
            else:
                x = block(x)

        x = self.base_model.norm(x)
        x = self.base_model.fc_norm(x)
        return {"features": x[:, 0]}
