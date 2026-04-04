from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()

Tensor = torch.Tensor


class ExpertMLPLoRA(nn.Module):
    """
    给共享 MLP 分支额外挂上的 expert 专属低秩增量。

    输入 z 的形状是 [B, K, M, D]：
    B: batch size
    K: 当前激活的 expert 数
    M: 每个 expert 的 token 数
    D: token embedding 维度
    """

    def __init__(self, num_experts: int, dim: int, rank: int, alpha: float = 1.0):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.num_experts = num_experts
        self.dim = dim
        self.rank = rank
        self.scale = alpha / rank

        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(dim, rank)) for _ in range(num_experts)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.empty(rank, dim)) for _ in range(num_experts)
        ])

        for a, b in zip(self.A, self.B):
            nn.init.kaiming_uniform_(a, a=5 ** 0.5)
            nn.init.zeros_(b)

    def forward(self, z: Tensor, expert_indices: Optional[Tensor] = None) -> Tensor:
        if z.ndim != 4:
            raise ValueError(f"z must be [B, K, M, D], got shape {tuple(z.shape)}")

        if expert_indices is None:
            if z.shape[1] != self.num_experts:
                raise ValueError(
                    f"expert_indices is required when K={z.shape[1]} != num_experts={self.num_experts}"
                )
            indices = range(self.num_experts)
        else:
            # 只取当前激活 expert 对应的低秩参数，避免无关 expert 参与前向。
            expert_indices = expert_indices.to(device=z.device, dtype=torch.long)
            indices = expert_indices.tolist()

        A = torch.stack([self.A[idx] for idx in indices], dim=0)
        B = torch.stack([self.B[idx] for idx in indices], dim=0)

        # LoRA 标准两步：先降维再升维。
        down = torch.einsum("bkmd,kdr->bkmr", z, A)
        up = torch.einsum("bkmr,krd->bkmd", down, B)
        return up * self.scale


class ExpertWrappedBlock(nn.Module):
    """
    一个 ViT block 的 SPiE 版本：
    注意力和主 MLP 是共享的，但 expert token 在 MLP 输出端会额外叠加 expert 专属 LoRA。
    """

    def __init__(
        self,
        base_block: nn.Module,
        num_experts: int,
        expert_tokens: int,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_tokens = expert_tokens

        self.norm1 = base_block.norm1
        self.attn = base_block.attn
        self._attn_accepts_mask = self._attention_accepts_mask(self.attn)
        self.ls1 = getattr(base_block, "ls1", nn.Identity())
        self.drop_path1 = getattr(base_block, "drop_path1", nn.Identity())

        self.norm2 = base_block.norm2
        self.mlp = base_block.mlp
        self.ls2 = getattr(base_block, "ls2", nn.Identity())
        self.drop_path2 = getattr(base_block, "drop_path2", nn.Identity())

        dim = getattr(self.norm1, "normalized_shape", None)
        if isinstance(dim, (tuple, list)):
            dim = int(dim[0])
        elif isinstance(dim, int):
            dim = dim
        else:
            dim = int(self.mlp.fc2.out_features)

        self.expert_lora = ExpertMLPLoRA(
            num_experts=num_experts,
            dim=dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )

    @staticmethod
    def _attention_accepts_mask(attn: nn.Module) -> bool:
        try:
            return "attn_mask" in inspect.signature(attn.forward).parameters
        except (TypeError, ValueError):
            return False

    def _forward_attention(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        if attn_mask is None:
            return self.attn(x)
        if self._attn_accepts_mask:
            return self.attn(x, attn_mask=attn_mask)
        # 一些 timm 版本的 attention.forward 不收 attn_mask，这里做兼容。
        return self._forward_attention_compat(x, attn_mask)

    def _forward_attention_compat(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        B, N, _ = x.shape
        num_heads = int(self.attn.num_heads)
        qkv = self.attn.qkv(x)
        head_dim = int(getattr(self.attn, "head_dim", qkv.shape[-1] // (3 * num_heads)))
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q_norm = getattr(self.attn, "q_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        k_norm = getattr(self.attn, "k_norm", None)
        if k_norm is not None:
            k = k_norm(k)

        attn_scores = (q * float(getattr(self.attn, "scale", head_dim ** -0.5))) @ k.transpose(-2, -1)
        attn_scores = attn_scores + attn_mask.to(device=attn_scores.device, dtype=attn_scores.dtype)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn.attn_drop(attn_probs)

        out = attn_probs @ v
        out = out.transpose(1, 2).reshape(B, N, int(getattr(self.attn, "attn_dim", num_heads * head_dim)))

        attn_norm = getattr(self.attn, "norm", None)
        if attn_norm is not None:
            out = attn_norm(out)

        out = self.attn.proj(out)
        out = self.attn.proj_drop(out)
        return out

    def forward(
        self,
        s: Tensor,
        attn_mask: Optional[Tensor],
        num_backbone_tokens: int,
        active_expert_indices: Optional[Tensor] = None,
    ) -> Tensor:
        if s.ndim != 3:
            raise ValueError(f"s must be [B, T, D], got shape {tuple(s.shape)}")

        # s 是把 backbone token 和 expert token 沿序列维拼起来后的结果。
        B, T, D = s.shape
        expert_tokens_total = T - num_backbone_tokens
        if expert_tokens_total < 0:
            raise ValueError("num_backbone_tokens cannot exceed the packed sequence length")
        if expert_tokens_total % self.expert_tokens != 0:
            raise ValueError(
                f"expert token segment ({expert_tokens_total}) is not divisible by expert_tokens ({self.expert_tokens})"
            )

        K = expert_tokens_total // self.expert_tokens

        # 注意力阶段是共享的，但会受 attn_mask 约束：
        # expert token 只能看 backbone token 和自己 expert 内部的 token。
        s = s + self.drop_path1(self.ls1(self._forward_attention(self.norm1(s), attn_mask=attn_mask)))

        s_norm = self.norm2(s)
        shared_delta = self.mlp(s_norm)

        if expert_tokens_total > 0:
            # 只取 expert token 那一段，应用 expert 专属 LoRA，然后再加回共享 MLP 输出。
            z_norm = s_norm[:, num_backbone_tokens:, :].reshape(B, K, self.expert_tokens, D)
            expert_delta = self.expert_lora(z_norm, expert_indices=active_expert_indices)
            shared_delta[:, num_backbone_tokens:, :] += expert_delta.reshape(B, K * self.expert_tokens, D)

        s = s + self.drop_path2(self.ls2(shared_delta))
        return s


@dataclass
class ExpertOutputs:
    # backbone 和 expert 分支的中间输出统一打包，便于训练和推理复用。
    cls_token: Optional[Tensor]
    backbone_tokens: Tensor
    expert_tokens: Tensor
    expert_pooled: Tensor
    active_expert_indices: Tensor


class ExpertViT(nn.Module):
    """
    共享 ViT 主干，加上隔离的 expert token 和 expert-only LoRA。
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_experts: int = 8,
        expert_tokens: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        use_expert_pos_embed: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be > 0, got {num_experts}")
        if expert_tokens <= 0:
            raise ValueError(f"expert_tokens must be > 0, got {expert_tokens}")

        try:
            import timm
        except ImportError as exc:
            raise ImportError("This module requires timm. Install it with `pip install timm`.") from exc

        base_model = timm.create_model(model_name, pretrained=pretrained)

        self.patch_embed = base_model.patch_embed
        self.cls_token = getattr(base_model, "cls_token", None)
        self.reg_token = getattr(base_model, "reg_token", None)
        self.pos_embed = getattr(base_model, "pos_embed", None)
        self.pos_drop = getattr(base_model, "pos_drop", nn.Identity())
        self.patch_drop = getattr(base_model, "patch_drop", nn.Identity())
        self.norm_pre = getattr(base_model, "norm_pre", nn.Identity())
        self.norm = getattr(base_model, "norm", nn.Identity())

        self.embed_dim = int(getattr(base_model, "embed_dim"))
        self.num_prefix_tokens = int(getattr(base_model, "num_prefix_tokens", 1))
        self.has_class_token = bool(getattr(base_model, "has_class_token", self.cls_token is not None))
        self.no_embed_class = bool(getattr(base_model, "no_embed_class", False))

        self.num_experts = num_experts
        self.expert_tokens = expert_tokens
        self.model_name = model_name

        # 每个 expert 拥有自己的一组可学习 token。
        self.expert_tokens_param = nn.ParameterList([
            nn.Parameter(torch.empty(1, expert_tokens, self.embed_dim))
            for _ in range(num_experts)
        ])
        for expert_tokens_param in self.expert_tokens_param:
            nn.init.trunc_normal_(expert_tokens_param, std=0.02)

        if use_expert_pos_embed:
            # expert token 也可以带独立的位置编码。
            self.expert_pos_embed = nn.ParameterList([
                nn.Parameter(torch.empty(1, expert_tokens, self.embed_dim))
                for _ in range(num_experts)
            ])
            for expert_pos_embed in self.expert_pos_embed:
                nn.init.trunc_normal_(expert_pos_embed, std=0.02)
        else:
            self.expert_pos_embed = None

        self.blocks = nn.ModuleList([
            ExpertWrappedBlock(
                base_block=block,
                num_experts=num_experts,
                expert_tokens=expert_tokens,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            for block in base_model.blocks
        ])

        self._mask_cache: Dict[Tuple[int, int, int, str, str], Tensor] = {}

        if freeze_backbone:
            self.freeze_backbone_()

    @torch.no_grad()
    def freeze_backbone_(self) -> None:
        # 只保留 expert 相关参数可训练，共享主干全部冻结。
        expert_ids = {id(p) for p in self.expert_parameters()}
        for p in self.parameters():
            p.requires_grad = id(p) in expert_ids

    def expert_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.expert_tokens_param
        if self.expert_pos_embed is not None:
            yield from self.expert_pos_embed
        for block in self.blocks:
            yield from block.expert_lora.A
            yield from block.expert_lora.B

    def expert_parameters_for_expert(self, expert_idx: int) -> Iterable[nn.Parameter]:
        if not (0 <= expert_idx < self.num_experts):
            raise ValueError(f"expert_idx must be in [0, {self.num_experts - 1}], got {expert_idx}")

        yield self.expert_tokens_param[expert_idx]
        if self.expert_pos_embed is not None:
            yield self.expert_pos_embed[expert_idx]
        for block in self.blocks:
            yield block.expert_lora.A[expert_idx]
            yield block.expert_lora.B[expert_idx]

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        expert_ids = {id(p) for p in self.expert_parameters()}
        for p in self.parameters():
            if id(p) not in expert_ids:
                yield p

    def _build_attn_mask(
        self,
        num_backbone_tokens: int,
        num_active_experts: int,
        expert_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        key = (num_backbone_tokens, num_active_experts, expert_tokens, str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        N = num_backbone_tokens
        K = num_active_experts
        M = expert_tokens
        T = N + K * M

        # 默认全部置为 -inf，表示不允许注意。
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        # backbone token 彼此之间完全可见。
        mask[:N, :N] = 0.0

        for k in range(K):
            q0 = N + k * M
            q1 = q0 + M
            # 第 k 个 expert token 允许看 backbone token。
            mask[q0:q1, :N] = 0.0
            # 第 k 个 expert token 允许看自己 expert 内部的 token。
            mask[q0:q1, q0:q1] = 0.0

        self._mask_cache[key] = mask
        return mask

    def _prepare_backbone_tokens(self, x: Tensor) -> Tensor:
        # 复用原始 ViT 的 patch embedding、prefix token 和位置编码逻辑。
        B = x.shape[0]
        x = self.patch_embed(x)

        prefix_tokens: List[Tensor] = []
        if self.cls_token is not None:
            prefix_tokens.append(self.cls_token.expand(B, -1, -1))
        if self.reg_token is not None:
            prefix_tokens.append(self.reg_token.expand(B, -1, -1))
        prefix = torch.cat(prefix_tokens, dim=1) if prefix_tokens else None

        if self.no_embed_class:
            if self.pos_embed is not None:
                if prefix is not None and self.num_prefix_tokens > 0:
                    prefix = prefix + self.pos_embed[:, :self.num_prefix_tokens, :]
                x = x + self.pos_embed[:, self.num_prefix_tokens:, :]
            if prefix is not None:
                x = torch.cat([prefix, x], dim=1)
        else:
            if prefix is not None:
                x = torch.cat([prefix, x], dim=1)
            if self.pos_embed is not None:
                x = x + self.pos_embed

        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return x

    def _select_experts(
        self,
        active_experts: Optional[Union[Sequence[int], Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        # active_experts=None 时表示推理时激活全部 expert。
        if active_experts is None:
            active_indices = torch.arange(self.num_experts, device=device, dtype=torch.long)
        elif isinstance(active_experts, torch.Tensor):
            active_indices = active_experts.to(device=device, dtype=torch.long)
        else:
            active_indices = torch.tensor(list(active_experts), device=device, dtype=torch.long)

        if active_indices.numel() == 0:
            raise ValueError("active_experts cannot be empty")
        if torch.any(active_indices < 0) or torch.any(active_indices >= self.num_experts):
            raise ValueError(f"active expert index out of range [0, {self.num_experts - 1}]")

        # 取出被激活 expert 的 token 参数。
        z = torch.stack([self.expert_tokens_param[idx] for idx in active_indices.tolist()], dim=1)
        if self.expert_pos_embed is not None:
            z = z + torch.stack([self.expert_pos_embed[idx] for idx in active_indices.tolist()], dim=1)
        return z, active_indices

    def forward_features(
        self,
        x: Tensor,
        active_experts: Optional[Union[Sequence[int], Tensor]] = None,
    ) -> ExpertOutputs:
        B = x.shape[0]
        x_backbone = self._prepare_backbone_tokens(x)
        num_backbone_tokens = x_backbone.shape[1]

        z, active_indices = self._select_experts(active_experts, device=x.device)
        z = z.expand(B, -1, -1, -1)
        K = z.shape[1]

        # 按 [backbone tokens, all active expert tokens] 的顺序打包成一个长序列。
        packed = torch.cat([
            x_backbone,
            z.reshape(B, K * self.expert_tokens, self.embed_dim),
        ], dim=1)

        attn_mask = self._build_attn_mask(
            num_backbone_tokens=num_backbone_tokens,
            num_active_experts=K,
            expert_tokens=self.expert_tokens,
            device=packed.device,
            dtype=packed.dtype,
        )

        for block in self.blocks:
            packed = block(
                packed,
                attn_mask=attn_mask,
                num_backbone_tokens=num_backbone_tokens,
                active_expert_indices=active_indices,
            )

        packed = self.norm(packed)
        backbone_tokens = packed[:, :num_backbone_tokens, :]
        expert_tokens = packed[:, num_backbone_tokens:, :].reshape(B, K, self.expert_tokens, self.embed_dim)
        # 一个 expert 的多个 token 取均值，得到该 expert 的 pooled 表示。
        expert_pooled = expert_tokens.mean(dim=2)
        cls_token = backbone_tokens[:, 0] if self.has_class_token else None

        return ExpertOutputs(
            cls_token=cls_token,
            backbone_tokens=backbone_tokens,
            expert_tokens=expert_tokens,
            expert_pooled=expert_pooled,
            active_expert_indices=active_indices,
        )

    def forward(
        self,
        x: Tensor,
        active_experts: Optional[Union[Sequence[int], Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[ExpertOutputs, Tensor]:
        out = self.forward_features(x, active_experts=active_experts)
        if return_dict:
            return out
        return out.expert_pooled

    def get_single_expert_state(self, expert_idx: int) -> Dict[str, Tensor]:
        # 导出单个 expert 的最小可迁移状态：expert token + 各层 expert LoRA。
        if not (0 <= expert_idx < self.num_experts):
            raise ValueError(f"expert_idx must be in [0, {self.num_experts - 1}], got {expert_idx}")

        state: Dict[str, Tensor] = {
            "expert_idx": torch.tensor(expert_idx, dtype=torch.long),
            "expert_tokens_param": self.expert_tokens_param[expert_idx].unsqueeze(1).detach().cpu(),
        }
        if self.expert_pos_embed is not None:
            state["expert_pos_embed"] = self.expert_pos_embed[expert_idx].unsqueeze(1).detach().cpu()

        for i, block in enumerate(self.blocks):
            state[f"blocks.{i}.expert_lora.A"] = block.expert_lora.A[expert_idx].unsqueeze(0).detach().cpu()
            state[f"blocks.{i}.expert_lora.B"] = block.expert_lora.B[expert_idx].unsqueeze(0).detach().cpu()
        return state

    @torch.no_grad()
    def load_single_expert_state(self, state: Dict[str, Tensor], target_expert_idx: int) -> None:
        # 把外部 expert 状态装载到当前模型指定 expert 槽位。
        if not (0 <= target_expert_idx < self.num_experts):
            raise ValueError(f"target_expert_idx must be in [0, {self.num_experts - 1}], got {target_expert_idx}")

        self.expert_tokens_param[target_expert_idx].copy_(
            state["expert_tokens_param"].squeeze(1).to(
                self.expert_tokens_param[target_expert_idx].device,
                self.expert_tokens_param[target_expert_idx].dtype,
            )
        )
        if self.expert_pos_embed is not None and "expert_pos_embed" in state:
            self.expert_pos_embed[target_expert_idx].copy_(
                state["expert_pos_embed"].squeeze(1).to(
                    self.expert_pos_embed[target_expert_idx].device,
                    self.expert_pos_embed[target_expert_idx].dtype,
                )
            )

        for i, block in enumerate(self.blocks):
            block.expert_lora.A[target_expert_idx].copy_(
                state[f"blocks.{i}.expert_lora.A"].squeeze(0).to(
                    block.expert_lora.A[target_expert_idx].device,
                    block.expert_lora.A[target_expert_idx].dtype,
                )
            )
            block.expert_lora.B[target_expert_idx].copy_(
                state[f"blocks.{i}.expert_lora.B"].squeeze(0).to(
                    block.expert_lora.B[target_expert_idx].device,
                    block.expert_lora.B[target_expert_idx].dtype,
                )
            )

    def save_single_expert(self, expert_idx: int, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_single_expert_state(expert_idx), path)

    def load_single_expert(self, path: Union[str, Path], target_expert_idx: int) -> None:
        state = torch.load(Path(path), map_location="cpu")
        self.load_single_expert_state(state, target_expert_idx=target_expert_idx)


class CosineClassifier(nn.Module):
    """
    单个 expert 对应的余弦分类头。
    """

    def __init__(self, dim: int, num_classes: int, init_scale: float = 10.0):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        self.dim = dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # 特征和权重都先归一化，再做缩放后的余弦相似度分类。
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = torch.exp(self.logit_scale).clamp(max=100.0)
        return scale * (x @ w.t())


class ExpertClassifierBank(nn.Module):
    """
    每个 expert / task 一个独立分类头。
    """

    def __init__(self, embed_dim: int, num_classes_per_expert: Sequence[int]):
        super().__init__()
        if len(num_classes_per_expert) == 0:
            raise ValueError("num_classes_per_expert cannot be empty")

        self.embed_dim = embed_dim
        self.num_experts = len(num_classes_per_expert)
        self.num_classes_per_expert = list(map(int, num_classes_per_expert))
        self.heads = nn.ModuleList([
            CosineClassifier(embed_dim, c) for c in self.num_classes_per_expert
        ])

    def forward_active(self, pooled: Tensor, active_expert_indices: Tensor) -> List[Tensor]:
        # pooled 的第 local_k 个位置，送入 active_expert_indices[local_k] 对应的 head。
        logits_per_expert: List[Tensor] = []
        for local_k, expert_idx in enumerate(active_expert_indices.tolist()):
            logits = self.heads[expert_idx](pooled[:, local_k, :])
            logits_per_expert.append(logits)
        return logits_per_expert

    def concat_active_logits(
        self,
        pooled: Tensor,
        active_expert_indices: Tensor,
    ) -> Tuple[Tensor, List[Dict[str, int]]]:
        # 把多个 expert 的局部 logits 拼接成一个大 logits，并返回位置映射表。
        logits_list = self.forward_active(pooled, active_expert_indices)
        all_logits = torch.cat(logits_list, dim=1)

        mapping: List[Dict[str, int]] = []
        for expert_idx in active_expert_indices.tolist():
            for local_class_idx in range(self.num_classes_per_expert[expert_idx]):
                mapping.append({
                    "expert_idx": int(expert_idx),
                    "local_class_idx": int(local_class_idx),
                })
        return all_logits, mapping

    def get_single_head_state(self, expert_idx: int) -> Dict[str, Tensor]:
        head = self.heads[expert_idx]
        return {
            "weight": head.weight.detach().cpu(),
            "logit_scale": head.logit_scale.detach().cpu(),
            "num_classes": torch.tensor(head.num_classes, dtype=torch.long),
        }

    @torch.no_grad()
    def load_single_head_state(self, state: Dict[str, Tensor], expert_idx: int) -> None:
        head = self.heads[expert_idx]
        if int(state["num_classes"].item()) != head.num_classes:
            raise ValueError(
                f"Head class count mismatch for expert {expert_idx}: "
                f"checkpoint has {int(state['num_classes'].item())}, current model expects {head.num_classes}"
            )
        head.weight.copy_(state["weight"].to(head.weight.device, head.weight.dtype))
        head.logit_scale.copy_(state["logit_scale"].to(head.logit_scale.device, head.logit_scale.dtype))


class IncrementalExpertModel(nn.Module):
    """
    SPiE 的完整模型：
    共享 expert backbone + 每个 expert 独立分类头。
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_experts: int = 8,
        num_classes_per_expert: Optional[Sequence[int]] = None,
        expert_tokens: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 1.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if num_classes_per_expert is None:
            raise ValueError("num_classes_per_expert must be provided")
        if len(num_classes_per_expert) != num_experts:
            raise ValueError(
                f"len(num_classes_per_expert) must equal num_experts; got {len(num_classes_per_expert)} vs {num_experts}"
            )

        self.backbone = ExpertViT(
            model_name=model_name,
            pretrained=pretrained,
            num_experts=num_experts,
            expert_tokens=expert_tokens,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            freeze_backbone=freeze_backbone,
        )
        self.heads = ExpertClassifierBank(
            embed_dim=self.backbone.embed_dim,
            num_classes_per_expert=num_classes_per_expert,
        )
        self.num_experts = num_experts
        self.num_classes_per_expert = list(map(int, num_classes_per_expert))

    def expert_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.backbone.expert_parameters()

    def expert_parameters_for_expert(self, expert_idx: int) -> Iterable[nn.Parameter]:
        yield from self.backbone.expert_parameters_for_expert(expert_idx)

    def head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.heads.parameters()

    def parameters_for_single_task(self, expert_idx: int) -> Iterable[nn.Parameter]:
        yield from self.backbone.expert_parameters_for_expert(expert_idx)
        yield from self.heads.heads[expert_idx].parameters()

    def parameter_groups_for_single_task(
        self,
        expert_idx: int,
        lr: float,
        head_lr: Optional[float] = None,
        expert_weight_decay: float = 0.0,
        head_weight_decay: float = 0.0,
    ) -> List[Dict[str, object]]:
        head_lr = lr if head_lr is None else head_lr
        return [
            {
                "params": list(self.backbone.expert_parameters_for_expert(expert_idx)),
                "lr": lr,
                "weight_decay": expert_weight_decay,
            },
            {
                "params": list(self.heads.heads[expert_idx].parameters()),
                "lr": head_lr,
                "weight_decay": head_weight_decay,
            },
        ]

    def forward_train_task(self, x: Tensor, expert_idx: int) -> Tensor:
        # 单任务训练：只激活一个 expert，并只走它自己的分类头。
        out = self.backbone(x, active_experts=[expert_idx], return_dict=True)
        pooled = out.expert_pooled[:, 0, :]
        logits = self.heads.heads[expert_idx](pooled)
        return logits

    @torch.no_grad()
    def predict_all(
        self,
        x: Tensor,
        active_experts: Optional[Union[Sequence[int], Tensor]] = None,
    ) -> Dict[str, Union[Tensor, List[Dict[str, int]]]]:
        self.eval()
        # 推理时可以同时激活多个 expert，再把所有局部类别拼成统一输出空间。
        out = self.backbone(x, active_experts=active_experts, return_dict=True)
        logits, mapping = self.heads.concat_active_logits(out.expert_pooled, out.active_expert_indices)
        pred_concat_idx = logits.argmax(dim=1)

        pred_expert_idx = torch.tensor(
            [mapping[i]["expert_idx"] for i in pred_concat_idx.tolist()],
            device=logits.device,
            dtype=torch.long,
        )
        pred_local_class_idx = torch.tensor(
            [mapping[i]["local_class_idx"] for i in pred_concat_idx.tolist()],
            device=logits.device,
            dtype=torch.long,
        )
        return {
            "logits": logits,
            "pred_concat_idx": pred_concat_idx,
            "pred_expert_idx": pred_expert_idx,
            "pred_local_class_idx": pred_local_class_idx,
            "mapping": mapping,
        }

    def get_single_expert_package(self, expert_idx: int) -> Dict[str, object]:
        return {
            "expert_state": self.backbone.get_single_expert_state(expert_idx),
            "head_state": self.heads.get_single_head_state(expert_idx),
            "expert_idx": expert_idx,
            "num_classes": self.num_classes_per_expert[expert_idx],
        }

    @torch.no_grad()
    def load_single_expert_package(self, package: Dict[str, object], target_expert_idx: int) -> None:
        self.backbone.load_single_expert_state(package["expert_state"], target_expert_idx)
        self.heads.load_single_head_state(package["head_state"], target_expert_idx)

    def save_single_expert_package(self, expert_idx: int, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_single_expert_package(expert_idx), path)

    def load_single_expert_package_from_file(self, path: Union[str, Path], target_expert_idx: int) -> None:
        package = torch.load(Path(path), map_location="cpu")
        self.load_single_expert_package(package, target_expert_idx)


def create_incremental_expert_vit(
    model_name: str,
    pretrained: bool = True,
    num_experts: int = 8,
    num_classes_per_expert: Optional[Sequence[int]] = None,
    expert_tokens: int = 4,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    freeze_backbone: bool = True,
) -> IncrementalExpertModel:
    return IncrementalExpertModel(
        model_name=model_name,
        pretrained=pretrained,
        num_experts=num_experts,
        num_classes_per_expert=num_classes_per_expert,
        expert_tokens=expert_tokens,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_backbone=freeze_backbone,
    )


def create_incremental_expert_vitb_224(
    pretrained: bool = True,
    num_experts: int = 8,
    num_classes_per_expert: Optional[Sequence[int]] = None,
    expert_tokens: int = 4,
    lora_rank: int = 8,
    lora_alpha: float = 1.0,
    freeze_backbone: bool = True,
) -> IncrementalExpertModel:
    return create_incremental_expert_vit(
        model_name="vit_base_patch16_224",
        pretrained=pretrained,
        num_experts=num_experts,
        num_classes_per_expert=num_classes_per_expert,
        expert_tokens=expert_tokens,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_backbone=freeze_backbone,
    )
