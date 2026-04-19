from typing import Optional

import torch
from torch import Tensor


class ARCLModAdam(torch.optim.Adam):
    def step(self, closure=None, update_factor_map: Optional[dict[int, Tensor]] = None):
        factor_map = update_factor_map or {}
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients")

                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0)
                        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1

                    if group["weight_decay"] != 0:
                        grad = grad.add(param, alpha=group["weight_decay"])

                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if group["amsgrad"]:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom_sq = max_exp_avg_sq
                    else:
                        denom_sq = exp_avg_sq

                    bias_correction1 = 1 - beta1 ** state["step"].item()
                    bias_correction2 = 1 - beta2 ** state["step"].item()
                    step_size = -group["lr"] / bias_correction1
                    denom = denom_sq.sqrt() / (bias_correction2 ** 0.5)
                    denom.add_(group["eps"])
                    update = exp_avg / denom

                    factor = factor_map.get(id(param))
                    if factor is not None:
                        update = update * factor

                    param.add_(update, alpha=step_size)

        return loss
