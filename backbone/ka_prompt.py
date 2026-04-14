import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PromptPool(nn.Module):
    def __init__(self, num_layers, pool_size, prompt_num, embed_dim, init_type="unif"):
        super().__init__()
        self.num_layers = int(num_layers)
        self.pool_size = int(pool_size)
        self.prompt_num = int(prompt_num)
        self.embed_dim = int(embed_dim)
        self.init_type = str(init_type)

        self.key_list = nn.ModuleList()
        self.prompt_list = nn.ModuleList()
        for _ in range(self.num_layers):
            layer_keys = nn.ParameterList(
                [self._init_key() for _ in range(self.pool_size)]
            )
            layer_prompts = nn.ParameterList(
                [self._init_prompt() for _ in range(self.pool_size)]
            )
            self.key_list.append(layer_keys)
            self.prompt_list.append(layer_prompts)

    def _init_key(self):
        key = nn.Parameter(torch.randn(self.embed_dim))
        if self.init_type == "unif":
            nn.init.uniform_(key, -1.0, 1.0)
        return key

    def _init_prompt(self):
        prompt = nn.Parameter(torch.randn(self.prompt_num, self.embed_dim))
        if self.init_type == "unif":
            nn.init.uniform_(prompt, -1.0, 1.0)
        return prompt

    def stack_keys(self, layer_idx, end=None):
        items = list(self.key_list[layer_idx][:end])
        return torch.stack(items, dim=0)

    def stack_prompts(self, layer_idx, end=None):
        items = list(self.prompt_list[layer_idx][:end])
        return torch.stack(items, dim=0)


class KAPromptBackbone(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        pool_size=20,
        prompt_num=8,
        gprompt_num=4,
        top_k=2,
        prompt_per_task=4,
        query_pos=-1,
        layer_g=(0, 1),
        layer_e=(2, 3, 4),
        init_type="unif",
        prompt_comp=True,
        fuse_prompt=True,
        use_prompt_mask=True,
    ):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.vit.out_dim = getattr(self.vit, "num_features", 768)
        self.out_dim = self.vit.out_dim

        self.num_layers = len(self.vit.blocks)
        self.pool_size = int(pool_size)
        self.prompt_num = int(prompt_num)
        self.gprompt_num = int(gprompt_num)
        self.top_k = int(top_k)
        self.prompt_per_task = int(prompt_per_task)
        self.query_pos = int(query_pos)
        self.layer_g = set(layer_g)
        self.layer_e = set(layer_e)
        self.prompt_comp = bool(prompt_comp)
        self.fuse_prompt = bool(fuse_prompt)
        self.use_prompt_mask = bool(use_prompt_mask)

        self.pool = PromptPool(
            num_layers=self.num_layers,
            pool_size=self.pool_size,
            prompt_num=self.prompt_num,
            embed_dim=self.out_dim,
            init_type=init_type,
        )
        self.general_prompt = nn.ParameterList(
            [nn.Parameter(torch.randn(self.gprompt_num, self.out_dim)) for _ in range(self.num_layers)]
        )
        for prompt in self.general_prompt:
            nn.init.uniform_(prompt, -1.0, 1.0)

        self.current_task = -1
        self.freeze_pretrained()

    def freeze_pretrained(self):
        self.vit.requires_grad_(False)
        self.vit.eval()

    def prompt_parameters(self):
        params = []
        for layer in self.pool.key_list:
            params.extend(list(layer.parameters()))
        for layer in self.pool.prompt_list:
            params.extend(list(layer.parameters()))
        params.extend(list(self.general_prompt.parameters()))
        return params

    def _current_prompt_slice(self, task_id):
        start = int(task_id) * self.prompt_per_task
        end = min(start + self.prompt_per_task, self.pool_size)
        return start, end

    def begin_task(self, task_id, train_loader=None):
        self.current_task = int(task_id)
        if self.current_task > 0 and train_loader is not None:
            self.greedy_init(train_loader, self.current_task)

    def extract_query(self, x):
        with torch.no_grad():
            representations = self.vit.forward_features(x)
        if representations.ndim == 2:
            return representations
        if self.query_pos >= 0:
            return representations[:, self.query_pos, :]
        return representations[:, 1:, :].mean(dim=1)

    def greedy_init(self, loader, task_id):
        start, end = self._current_prompt_slice(task_id)
        if start <= 0 or start >= end:
            return

        all_keys = self.pool.stack_keys(0, end=start).to(next(self.parameters()).device)
        all_keys = F.normalize(all_keys, dim=-1)
        sim_scores = []
        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.to(next(self.parameters()).device)
                query = F.normalize(self.extract_query(inputs), dim=-1)
                sim_scores.append(all_keys @ query.T)
        sim_scores = (torch.cat(sim_scores, dim=1) + 1.0) / 2.0

        selected_ids = []
        real_num = 0
        num_to_init = end - start
        for _ in range(num_to_init):
            if real_num > 0:
                selected_scores = sim_scores[selected_ids[:real_num]]
                if selected_scores.ndim > 1 and selected_scores.shape[0] > 1:
                    selected_scores = selected_scores.max(0)[0].unsqueeze(0)
                overall_weight = (sim_scores - selected_scores).clamp(min=0.0).sum(1)
            else:
                overall_weight = sim_scores.sum(1)

            if torch.all(overall_weight == 0):
                two_id = selected_ids[:real_num] if real_num > 0 else [0]
                picked = two_id[:2] if len(two_id) >= 2 else [two_id[0], two_id[0]]
                selected_ids.append(picked)
            else:
                selected_ids.append(torch.argmax(overall_weight).item())
                real_num += 1

        for layer_idx in range(self.num_layers):
            for offset, src in enumerate(selected_ids):
                dst = start + offset
                if isinstance(src, list):
                    prompt = (
                        self.pool.prompt_list[layer_idx][src[0]].detach()
                        + self.pool.prompt_list[layer_idx][src[1]].detach()
                    ) / 2.0
                    key = (
                        self.pool.key_list[layer_idx][src[0]].detach()
                        + self.pool.key_list[layer_idx][src[1]].detach()
                    ) / 2.0
                else:
                    prompt = self.pool.prompt_list[layer_idx][src].detach().clone()
                    key = self.pool.key_list[layer_idx][src].detach().clone()
                self.pool.prompt_list[layer_idx][dst].data.copy_(prompt)
                self.pool.key_list[layer_idx][dst].data.copy_(key)

    def similarity(self, query, task_id=None, training=None):
        if training is None:
            training = self.training
        key_bank = self.pool.stack_keys(0).to(query.device)
        q = F.normalize(query, dim=-1)
        k = F.normalize(key_bank, dim=-1)
        sim = torch.matmul(q, k.T)
        dist = 1.0 - sim

        if self.use_prompt_mask and task_id is not None and training:
            start, end = self._current_prompt_slice(task_id)
            masked = torch.full_like(dist, 2.0)
            masked[:, start:end] = dist[:, start:end]
            dist = masked

        val, idx = torch.topk(dist, self.top_k, dim=1, largest=False)
        return val, idx

    def get_prompts(self, layer_idx, keys):
        batch_size = keys.shape[0]
        if layer_idx in self.layer_g:
            return self.general_prompt[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)

        if layer_idx in self.layer_e:
            prompt_tensor = self.pool.stack_prompts(layer_idx).to(keys.device)
            prompts = prompt_tensor[keys]
            _, top_k, prompt_len, embed_dim = prompts.shape

            if self.fuse_prompt:
                return prompts.mean(dim=1)
            if self.prompt_comp:
                half = prompt_len // 2
                prompt_k = prompts[:, :, :half, :].reshape(batch_size, top_k * half, embed_dim)
                prompt_v = prompts[:, :, half:, :].reshape(batch_size, top_k * (prompt_len - half), embed_dim)
                return torch.cat((prompt_v, prompt_k), dim=1)
            return prompts.reshape(batch_size, top_k * prompt_len, embed_dim)

        return None

    def _forward_attn_with_prompts(self, block, x, prompts):
        attn = block.attn
        batch_size, seq_len, embed_dim = x.shape

        qkv_q = attn.qkv(x).reshape(batch_size, seq_len, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, _, _ = qkv_q.unbind(0)
        half = prompts.shape[1] // 2

        prompt_k_input = torch.cat([x[:, :1, :], prompts[:, :half, :], x[:, 1:, :]], dim=1)
        prompt_v_input = torch.cat([x[:, :1, :], prompts[:, half:, :], x[:, 1:, :]], dim=1)

        qkv_k = attn.qkv(prompt_k_input).reshape(batch_size, -1, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        qkv_v = attn.qkv(prompt_v_input).reshape(batch_size, -1, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        _, k, _ = qkv_k.unbind(0)
        _, _, v = qkv_v.unbind(0)

        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q = q * attn.scale
        attn_score = q @ k.transpose(-2, -1)
        attn_score = attn_score.softmax(dim=-1)
        attn_score = attn.attn_drop(attn_score)
        out = attn_score @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out

    def _forward_block_with_prompts(self, block, x, prompts):
        x = x + block.drop_path1(block.ls1(self._forward_attn_with_prompts(block, block.norm1(x), prompts)))
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def _encode_with_keys(self, x, keys):
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        for layer_idx, block in enumerate(self.vit.blocks):
            prompts = self.get_prompts(layer_idx, keys)
            if prompts is not None:
                x = self._forward_block_with_prompts(block, x, prompts.to(x.device))
            else:
                x = block(x)

        x = self.vit.norm(x)
        cls_features = x[:, 0, :]
        return cls_features

    def forward_with_keys(self, x, keys):
        cls_features = self._encode_with_keys(x, keys)
        return {"features": cls_features, "cls_features": cls_features}

    def forward_old_prompt_mix(self, x, task_id, new_keys, old_backbone, tau=0.01):
        if task_id <= 0:
            return self.forward_with_keys(x, new_keys)

        query = self.extract_query(x)
        old_key_bank = old_backbone.pool.stack_keys(0, end=task_id * self.prompt_per_task).to(query.device)
        old_key_bank = F.normalize(old_key_bank, dim=-1)

        selected_new_keys = self.pool.stack_keys(0).to(query.device)[new_keys]
        selected_new_keys = F.normalize(selected_new_keys, dim=-1)
        query_norm = F.normalize(query, dim=-1)

        old_scores = torch.matmul(query_norm, old_key_bank.T)
        new_scores = torch.matmul(selected_new_keys, query_norm.unsqueeze(-1)).squeeze(-1)
        score_w = new_scores.min(dim=-1, keepdim=True)[0]
        val, idx = torch.topk(old_scores, self.top_k, dim=1, largest=True)
        prompt_weight_old = torch.exp((val - score_w).clamp(max=0.0) / max(tau, 1e-6))
        prompt_weight = torch.cat((torch.ones_like(prompt_weight_old), prompt_weight_old), dim=1).unsqueeze(-1).unsqueeze(-1)

        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        for layer_idx, block in enumerate(self.vit.blocks):
            prompts = None
            if layer_idx in self.layer_g:
                prompts = self.general_prompt[layer_idx].unsqueeze(0).expand(query.shape[0], -1, -1)
            elif layer_idx in self.layer_e:
                prompt_tensor = self.pool.stack_prompts(layer_idx).to(query.device)
                prompts_new = prompt_tensor[new_keys]
                prompts_old = old_backbone.pool.stack_prompts(
                    layer_idx,
                    end=task_id * self.prompt_per_task,
                ).to(query.device)[idx]
                merged = torch.cat((prompts_new, prompts_old), dim=1)
                prompts = (merged * prompt_weight).sum(1) / prompt_weight.sum(1).clamp_min(1e-6)

            if prompts is not None:
                x = self._forward_block_with_prompts(block, x, prompts.to(x.device))
            else:
                x = block(x)

        x = self.vit.norm(x)
        cls_features = x[:, 0, :]
        return {"features": cls_features, "cls_features": cls_features}

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        if fc_only:
            return {"features": x, "cls_features": x}

        task_id = self.current_task if adapter_id < 0 else adapter_id
        if task_id < 0:
            query = self.extract_query(x)
            _, keys = self.similarity(query, task_id=0, training=False)
        else:
            query = self.extract_query(x)
            distance, keys = self.similarity(query, task_id=task_id, training=train)
        res = self.forward_with_keys(x, keys)
        res["query"] = query
        res["keys"] = keys
        if task_id >= 0:
            res["distance"] = distance
        return res
