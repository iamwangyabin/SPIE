import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()
import timm


class MQMKPromptPool(nn.Module):
    def __init__(
        self,
        pool_size,
        prompt_length,
        embed_dim,
        num_heads,
        top_k,
        prompt_key=True,
        prompt_key_init="uniform",
        prompt_init="uniform",
        embedding_key="cls",
        batchwise_prompt=False,
        use_prompt_mask=True,
        multi_query=False,
        multi_key=False,
        class_per_task=20,
        k_key=1,
        class_group=1,
        num_layers=1,
    ):
        super().__init__()
        self.pool_size = int(pool_size)
        self.prompt_length = int(prompt_length)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.top_k = int(top_k)
        self.prompt_key = bool(prompt_key)
        self.embedding_key = embedding_key
        self.batchwise_prompt = bool(batchwise_prompt)
        self.use_prompt_mask = bool(use_prompt_mask)
        self.multi_query = bool(multi_query)
        self.multi_key = bool(multi_key)
        self.class_per_task = int(class_per_task)
        self.k_key = int(k_key)
        self.class_group = int(class_group)
        self.group_key_num = int(math.ceil(self.class_per_task / max(self.class_group, 1)))
        self.num_layers = int(num_layers)

        prompt_shape = (
            self.num_layers,
            2,
            self.pool_size,
            self.prompt_length,
            self.num_heads,
            self.head_dim,
        )
        self.prompt = nn.Parameter(torch.randn(prompt_shape))
        if prompt_init == "uniform":
            nn.init.uniform_(self.prompt, -1.0, 1.0)
        elif prompt_init == "zero":
            nn.init.zeros_(self.prompt)

        if self.prompt_key:
            if self.multi_key:
                key_classes = self.group_key_num if self.class_group > 1 else self.class_per_task
                key_shape = (self.pool_size, key_classes, self.embed_dim)
            else:
                key_shape = (self.pool_size, self.embed_dim)
            self.prompt_key_param = nn.Parameter(torch.randn(key_shape))
            if prompt_key_init == "uniform":
                nn.init.uniform_(self.prompt_key_param, -1.0, 1.0)
            elif prompt_key_init == "zero":
                nn.init.zeros_(self.prompt_key_param)
        else:
            prompt_mean = torch.mean(self.prompt, dim=(0, 1, 3, 4))
            self.register_buffer("prompt_key_param", prompt_mean)

    @staticmethod
    def _normalize(x, dim=-1, eps=1e-12):
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        inv_norm = torch.rsqrt(torch.clamp(square_sum, min=eps))
        return x * inv_norm

    def _select_embedding(self, x_embed, cls_features=None):
        if self.embedding_key == "mean":
            return torch.mean(x_embed, dim=1)
        if self.embedding_key == "max":
            return torch.max(x_embed, dim=1)[0]
        if self.embedding_key == "mean_max":
            return torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        if self.embedding_key == "cls":
            if cls_features is not None:
                return cls_features
            return torch.max(x_embed, dim=1)[0]
        raise NotImplementedError(f"Unsupported embedding key: {self.embedding_key}")

    def forward(
        self,
        x_embed,
        task_id,
        cls_features=None,
        train=False,
        query=False,
        target=None,
        prompt_mask=None,
    ):
        x_embed_key = self._select_embedding(x_embed, cls_features=cls_features)
        prompt_key_norm = self._normalize(self.prompt_key_param, dim=-1)
        x_embed_norm = self._normalize(x_embed_key, dim=-1)
        use_multi_query = self.multi_query and x_embed_norm.ndim == 3 and not query

        if not use_multi_query:
            if not self.multi_key:
                similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()).t()
            else:
                batch_size = x_embed_norm.shape[0]
                pool_size = prompt_key_norm.shape[0]
                prompt_key_expand = prompt_key_norm.unsqueeze(0).expand(
                    batch_size, pool_size, prompt_key_norm.shape[1], self.embed_dim
                )
                x_embed_expand = x_embed_norm.unsqueeze(1).unsqueeze(1).expand_as(prompt_key_expand)
                similarity = torch.sum(prompt_key_expand * x_embed_expand, dim=-1)
                if target is None:
                    similarity = torch.topk(similarity, dim=-1, k=self.k_key)[0].sum(dim=-1)
                else:
                    class_index = target % self.class_per_task
                    similarity = similarity[
                        torch.arange(batch_size, device=target.device).unsqueeze(1),
                        :,
                        class_index.unsqueeze(1),
                    ].squeeze(1)
        else:
            if not self.multi_key:
                batch_size = x_embed_norm.shape[0]
                prompt_key_expand = prompt_key_norm.unsqueeze(0).expand(batch_size, -1, -1)
                similarity = torch.sum(prompt_key_expand * x_embed_norm, dim=-1)
            else:
                batch_size = x_embed_norm.shape[0]
                key_classes = prompt_key_norm.shape[1]
                prompt_key_expand = prompt_key_norm.unsqueeze(0).expand(
                    batch_size, self.pool_size, key_classes, self.embed_dim
                )
                x_embed_expand = x_embed_norm.unsqueeze(2).expand(batch_size, self.pool_size, key_classes, self.embed_dim)
                similarity = torch.sum(prompt_key_expand * x_embed_expand, dim=-1)
                if target is None:
                    similarity = torch.topk(similarity, dim=-1, k=self.k_key)[0].sum(dim=-1)
                else:
                    if self.class_group > 1:
                        class_index = (target % self.class_per_task) // self.class_group
                    else:
                        class_index = target % self.class_per_task
                    similarity = similarity[
                        torch.arange(batch_size, device=target.device).unsqueeze(1),
                        :,
                        class_index.unsqueeze(1),
                    ].squeeze(1)

        _, idx = torch.topk(similarity, k=self.top_k, dim=-1)

        if self.batchwise_prompt:
            prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            if prompt_id.shape[0] < self.pool_size:
                fill_prompt = torch.min(idx.flatten())
                prompt_id = torch.cat(
                    [
                        prompt_id,
                        torch.full(
                            (self.pool_size - prompt_id.shape[0],),
                            fill_prompt,
                            device=prompt_id.device,
                        ),
                    ]
                )
                id_counts = torch.cat(
                    [
                        id_counts,
                        torch.zeros(self.pool_size - id_counts.shape[0], device=id_counts.device, dtype=id_counts.dtype),
                    ]
                )
            _, major_idx = torch.topk(id_counts, k=self.top_k)
            idx = prompt_id[major_idx].expand(x_embed.shape[0], -1).contiguous()

        if prompt_mask is not None:
            idx = prompt_mask

        batched_prompt_raw = self.prompt[:, :, idx]
        num_layers, dual, batch_size, top_k, prompt_length, num_heads, head_dim = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(
            num_layers,
            batch_size,
            dual,
            top_k * prompt_length,
            num_heads,
            head_dim,
        )

        if self.multi_key:
            selected_key = prompt_key_norm[idx]
        else:
            selected_key = prompt_key_norm[idx]

        if query:
            reduce_sim = torch.zeros((), device=x_embed.device)
        else:
            sim = similarity.clone()
            if prompt_mask is not None:
                sim = sim[:, task_id]
            else:
                sim[:, task_id + 1 :] = -1
            reduce_sim = torch.sum(sim) / x_embed.shape[0]

        return {
            "batched_prompt": batched_prompt,
            "prompt_idx": idx,
            "selected_key": selected_key,
            "prompt_key_norm": prompt_key_norm,
            "x_embed_norm": x_embed_norm,
            "similarity": similarity,
            "reduce_sim": reduce_sim,
        }


class MQMKBackbone(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        pool_size=10,
        prompt_length=90,
        top_k=1,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_key=True,
        prompt_key_init="uniform",
        batchwise_prompt=False,
        use_prompt_mask=True,
        use_g_prompt=True,
        g_prompt_length=5,
        g_prompt_layer_idx=(0, 1),
        use_e_prompt=True,
        e_prompt_layer_idx=(2, 3, 4),
        same_key_value=False,
        multi_query=True,
        multi_key=True,
        k_key=1,
        class_group=1,
        class_per_task=20,
        perfect_match=False,
    ):
        super().__init__()
        self.base_model_name = model_name
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit.out_dim = getattr(self.vit, "num_features", 768)
        self.out_dim = self.vit.out_dim
        self.current_task = -1
        self.top_k = int(top_k)
        self.pool_size = int(pool_size)
        self.use_prompt_mask = bool(use_prompt_mask)
        self.multi_query = bool(multi_query)
        self.multi_key = bool(multi_key)
        self.use_g_prompt = bool(use_g_prompt)
        self.use_e_prompt = bool(use_e_prompt)
        self.g_prompt_layer_idx = list(g_prompt_layer_idx or [])
        self.e_prompt_layer_idx = list(e_prompt_layer_idx or [])
        self.perfect_match = bool(perfect_match)

        num_heads = self.vit.blocks[0].attn.num_heads
        head_dim = self.out_dim // num_heads

        num_g_layers = len(self.g_prompt_layer_idx)
        if self.use_g_prompt and num_g_layers > 0 and int(g_prompt_length) > 0:
            g_shape = (num_g_layers, 2, int(g_prompt_length), num_heads, head_dim)
            self.g_prompt = nn.Parameter(torch.randn(g_shape))
            if prompt_init == "uniform":
                nn.init.uniform_(self.g_prompt, -1.0, 1.0)
            elif prompt_init == "zero":
                nn.init.zeros_(self.g_prompt)
            if same_key_value:
                self.g_prompt.data[:, 1].copy_(self.g_prompt.data[:, 0])
        else:
            self.g_prompt = None

        self.e_prompt = None
        if self.use_e_prompt and len(self.e_prompt_layer_idx) > 0:
            self.e_prompt = MQMKPromptPool(
                pool_size=pool_size,
                prompt_length=prompt_length,
                embed_dim=self.out_dim,
                num_heads=num_heads,
                top_k=top_k,
                prompt_key=prompt_key,
                prompt_key_init=prompt_key_init,
                prompt_init=prompt_init,
                embedding_key=embedding_key,
                batchwise_prompt=batchwise_prompt,
                use_prompt_mask=use_prompt_mask,
                multi_query=multi_query,
                multi_key=multi_key,
                class_per_task=class_per_task,
                k_key=k_key,
                class_group=class_group,
                num_layers=len(self.e_prompt_layer_idx),
            )

        self.vit.requires_grad_(False)
        self.vit.eval()

    def begin_task(self, task_id, train_loader=None):
        self.current_task = int(task_id)

    def prompt_parameters(self):
        params = []
        if self.g_prompt is not None:
            params.append(self.g_prompt)
        if self.e_prompt is not None:
            params.extend(list(self.e_prompt.parameters()))
        return params

    def extract_query(self, x):
        with torch.no_grad():
            representations = self.vit.forward_features(x)
        if representations.ndim == 2:
            return representations
        return representations[:, 0, :]

    def _build_prompt_mask(self, batch_size, task_id, device, training, query):
        if self.e_prompt is None or not self.use_prompt_mask:
            return None
        if not (training or self.perfect_match or query):
            return None

        start = task_id * self.top_k
        end = (task_id + 1) * self.top_k
        if end > self.pool_size:
            return None
        single_prompt_mask = torch.arange(start, end, device=device)
        return single_prompt_mask.unsqueeze(0).expand(batch_size, -1)

    def _prefix_attention(self, block, x, prompt):
        attn = block.attn
        batch_size, seq_len, embed_dim = x.shape
        qkv = attn.qkv(x).reshape(batch_size, seq_len, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)

        if prompt is not None:
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous()
            key_prefix = prompt[0]
            value_prefix = prompt[1]
            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        q = q * attn.scale
        attn_score = q @ k.transpose(-2, -1)
        attn_score = attn_score.softmax(dim=-1)
        attn_score = attn.attn_drop(attn_score)
        out = attn_score @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out

    def _forward_block(self, block, x, prompt):
        x = x + block.drop_path1(block.ls1(self._prefix_attention(block, block.norm1(x), prompt)))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def forward(
        self,
        x,
        adapter_id=-1,
        train=False,
        fc_only=False,
        cls_features=None,
        target=None,
        query=False,
    ):
        if fc_only:
            return {"features": x, "cls_features": x}

        task_id = self.current_task if adapter_id < 0 else int(adapter_id)
        prompt_res = {}
        prompt_mask = None

        x_tokens = self.vit.patch_embed(x)
        x_tokens = self.vit._pos_embed(x_tokens)
        x_tokens = self.vit.patch_drop(x_tokens)
        x_tokens = self.vit.norm_pre(x_tokens)

        if self.e_prompt is not None:
            prompt_mask = self._build_prompt_mask(x_tokens.shape[0], task_id, x_tokens.device, train, query)
            prompt_res = self.e_prompt(
                x_tokens,
                task_id=task_id,
                cls_features=cls_features,
                train=train,
                query=query,
                target=target,
                prompt_mask=prompt_mask,
            )
            e_prompt = prompt_res["batched_prompt"]
        else:
            e_prompt = None

        g_prompt_counter = -1
        e_prompt_counter = -1
        for layer_idx, block in enumerate(self.vit.blocks):
            layer_prompt = None
            if self.g_prompt is not None and layer_idx in self.g_prompt_layer_idx:
                g_prompt_counter += 1
                layer_prompt = self.g_prompt[g_prompt_counter].unsqueeze(0).expand(x_tokens.shape[0], -1, -1, -1, -1)
            elif e_prompt is not None and layer_idx in self.e_prompt_layer_idx:
                e_prompt_counter += 1
                layer_prompt = e_prompt[e_prompt_counter]
            x_tokens = self._forward_block(block, x_tokens, layer_prompt)

        x_tokens = self.vit.norm(x_tokens)
        cls_token = x_tokens[:, 0, :]
        return {
            "features": cls_token,
            "cls_features": cls_token,
            "prompt_idx": prompt_res.get("prompt_idx"),
            "selected_key": prompt_res.get("selected_key"),
            "reduce_sim": prompt_res.get("reduce_sim"),
            "similarity": prompt_res.get("similarity"),
        }
