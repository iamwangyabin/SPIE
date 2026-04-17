import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """PyTorch implementation of Lion (EvoLved Sign Momentum)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                param.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
