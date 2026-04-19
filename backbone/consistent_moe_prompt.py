import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()
import timm


class IntermReader(nn.Module):
    def __init__(self, dst_param_id: Optional[int], module_name: str):
        super().__init__()
        self.dst_param_id = dst_param_id
        self.module_name = module_name

    def forward(self, x: Tensor) -> Tensor:
        return x


class SparseDispatcher:
    def __init__(self, num_experts: int, gates: Tensor):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp: Tensor) -> List[Tensor]:
        inp_exp = inp[self._batch_index].squeeze(1)
        return list(torch.split(inp_exp, self._part_sizes, dim=0))

    def combine(self, expert_out: List[Tensor], multiply_by_gates: bool = True) -> Tensor:
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            requires_grad=True,
            device=stitched.device,
            dtype=stitched.dtype,
        )
        return zeros.index_add(0, self._batch_index, stitched)


class MoE(nn.Module):
    def __init__(self, input_size: int, num_experts: int, topk: int, noisy_gating: bool):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.topk = topk
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        if self.noisy_gating:
            self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.topk <= self.num_experts

    def cv_squared(self, x: Tensor) -> Tensor:
        eps = torch.finfo(self.w_gate.dtype).eps
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates: Tensor) -> Tensor:
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self,
        clean_values: Tensor,
        noisy_values: Tensor,
        noise_stddev: Tensor,
        noisy_top_values: Tensor,
    ) -> Tensor:
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.topk
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x: Tensor, train: bool, noise_epsilon: float = 1e-2) -> Tuple[Tensor, Tensor]:
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            noisy_logits = clean_logits
            noise_stddev = None
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.topk + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.topk]
        top_k_indices = top_indices[:, : self.topk]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.topk < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


class MoEPrompt(MoE):
    def __init__(
        self,
        embed_dim: int,
        prompt_len: int,
        num_experts: int,
        topk: int,
        noisy_gating: bool,
    ):
        super().__init__(embed_dim, num_experts, topk, noisy_gating)
        self.prompt_len = prompt_len
        self.embed_dim = embed_dim
        self.experts = nn.Parameter(torch.randn(num_experts, prompt_len, embed_dim))
        nn.init.uniform_(self.experts, -1, 1)
        self.interm_reader_loss = IntermReader(None, "interm_reader_loss")
        self.interm_reader_1 = IntermReader(id(self.w_gate), "interm_reader_1")
        self.interm_reader_2 = IntermReader(id(self.experts), "interm_reader_2")

    def forward(self, x: Tensor, q_x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.training:
            q_x = self.interm_reader_1(q_x)

        gates, load = self.noisy_top_k_gating(q_x, self.training)
        if not self.training:
            gates = self.interm_reader_2(gates)

        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        self.interm_reader_loss(balance_loss)

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(q_x)
        expert_outputs = [
            self.experts[i].reshape(1, -1).expand(expert_inputs[i].shape[0], -1)
            for i in range(self.num_experts)
        ]
        batch_prompts = dispatcher.combine(expert_outputs)
        batch_prompts = batch_prompts.view(batch_prompts.shape[0], self.prompt_len, self.embed_dim)
        return batch_prompts, balance_loss


class ConsistentMoEPromptBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.augreg_in21k",
        pretrained: bool = True,
        prompt_len: int = 4,
        prompt_start_block: int = 0,
        prompt_end_block: int = 11,
        moe_num_experts: int = 36,
        moe_topk: int = 16,
        moe_noisy_gating: bool = True,
        moe_frozen_qx_coef: float = 0.0,
    ):
        super().__init__()
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = getattr(self.base_model, "num_features", 768)
        self.pretrained_cfg = getattr(self.base_model, "pretrained_cfg", {})
        self.prompt_len = prompt_len
        self.prompt_start_block = prompt_start_block
        self.prompt_end_block = prompt_end_block
        self.moe_frozen_qx_coef = moe_frozen_qx_coef

        self.prompt_layers = nn.ModuleDict()
        if self.prompt_len > 0:
            for block_idx in range(prompt_start_block, prompt_end_block + 1):
                self.prompt_layers[str(block_idx)] = MoEPrompt(
                    embed_dim=self.out_dim,
                    prompt_len=prompt_len,
                    num_experts=moe_num_experts,
                    topk=moe_topk,
                    noisy_gating=moe_noisy_gating,
                )

    def get_trainable_named_parameters(self):
        for name, param in self.named_parameters():
            yield name, param

    def _forward_tokens(self, x: Tensor) -> Tensor:
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)
        x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)
        return x

    def forward(self, x, adapter_id=None, train: bool = False) -> Dict[str, Tensor]:
        if isinstance(x, (tuple, list)):
            images, frozen_qx = x
        else:
            images, frozen_qx = x, None

        x = self._forward_tokens(images)
        moe_losses: List[Tensor] = []

        for block_idx, block in enumerate(self.base_model.blocks):
            prompt_layer = self.prompt_layers[str(block_idx)] if str(block_idx) in self.prompt_layers else None
            if prompt_layer is not None:
                src_num_tokens = x.shape[1]
                q_x = x[:, 0]
                if frozen_qx is not None:
                    q_x = self.moe_frozen_qx_coef * frozen_qx + (1.0 - self.moe_frozen_qx_coef) * q_x
                prompts, balance_loss = prompt_layer(x, q_x)
                moe_losses.append(balance_loss)
                x_prompt = torch.cat([x, prompts], dim=1)
                x_attn = block.attn(block.norm1(x_prompt))[:, :src_num_tokens]
                x = x + block.drop_path1(block.ls1(x_attn))
                x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            else:
                x = block(x)

        x = self.base_model.norm(x)
        x = self.base_model.fc_norm(x)
        features = x[:, 0]
        moe_loss = None
        if moe_losses:
            moe_loss = torch.stack(moe_losses).mean()

        return {"features": features, "moe_loss": moe_loss}
