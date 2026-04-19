import math

import torch
from torch import Tensor
from torch.optim import Optimizer


def relaxed_proj(proj: Tensor, alpha: float) -> Tensor:
    if alpha == 1.0:
        return proj
    eye = torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)
    return alpha * proj + (1.0 - alpha) * eye


class ProjectedModAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.update_projection_dict = {}
        self.mask_dict = {}
        self.null_alpha1 = 1.0
        self.null_alpha2 = 1.0
        self.impt_more_relax = 0.03
        self.impt_lr_decay = 1.0

    def _project_update(self, param: Tensor, update: Tensor) -> Tensor:
        pid = id(param)
        proj_dict = self.update_projection_dict.get(pid)
        if proj_dict is None:
            return update
        if update.ndim != 3 or update.shape[0] != 1:
            return update

        src = update.squeeze(0)
        proj1 = proj_dict.get("interm_reader_1")
        proj2 = proj_dict.get("interm_reader_2")
        if proj1 is None or proj2 is None:
            return update

        proj1 = proj1.to(device=src.device, dtype=src.dtype)
        proj2 = proj2.to(device=src.device, dtype=src.dtype)

        if pid in self.mask_dict:
            mask_intersection, mask_nonunionset, mask_only_old, mask_only_new = self.mask_dict[pid]
            mask_intersection = mask_intersection.to(device=src.device, dtype=src.dtype)
            mask_nonunionset = mask_nonunionset.to(device=src.device, dtype=src.dtype)
            mask_only_old = mask_only_old.to(device=src.device, dtype=src.dtype)
            mask_only_new = mask_only_new.to(device=src.device, dtype=src.dtype)

            null_alpha = self.null_alpha1
            relaxed_alpha = max(null_alpha - self.impt_more_relax, 0.0)
            up_only_old = torch.zeros_like(src) * mask_only_old
            up_only_new = src * self.impt_lr_decay * mask_only_new
            up_intersection = (relaxed_proj(proj2, null_alpha) @ src @ relaxed_proj(proj1, null_alpha)) * mask_intersection
            up_nonunionset = (relaxed_proj(proj2, relaxed_alpha) @ src @ relaxed_proj(proj1, relaxed_alpha)) * mask_nonunionset
            src = up_only_old + up_only_new + up_intersection + up_nonunionset
        else:
            src = relaxed_proj(proj2, self.null_alpha2) @ src @ relaxed_proj(proj1, self.null_alpha1)
        return src.unsqueeze(0)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if weight_decay != 0:
                    grad = grad.add(p.detach(), alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = (-lr / bias_correction1) * exp_avg / denom
                update = self._project_update(p, update)

                with torch.no_grad():
                    p.add_(update)
        return loss
