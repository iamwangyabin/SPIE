import math

import torch
import torch.nn as nn

from backbone.vit_min import VisionTransformer as MiNVisionTransformer

__all__ = ["VisionTransformer"]


class ResidualAdapterTaskBranch(nn.Module):
    """Task-specific full-size adapter with the same parameter count as MiN mu+sigmma."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PiResidualAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=384, device=None):
        super().__init__()

        self.MLP = nn.Linear(in_dim, out_dim)
        torch.nn.init.constant_(self.MLP.weight, 0)
        torch.nn.init.constant_(self.MLP.bias, 0)

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        factory_kwargs = {"device": device, "dtype": torch.float32}

        self.hidden_dim = hidden_dim

        self.register_buffer("w_down", torch.empty((in_dim, self.hidden_dim), **factory_kwargs))
        nn.init.kaiming_uniform_(self.w_down, a=math.sqrt(5))

        self.task_adapters = nn.ModuleList()

        self.register_buffer("w_up", torch.empty((self.hidden_dim, out_dim), **factory_kwargs))
        nn.init.kaiming_uniform_(self.w_up, a=math.sqrt(5))

        self.weight_noise = None

    def update_noise(self):
        self.task_adapters.append(ResidualAdapterTaskBranch(self.hidden_dim))

    def init_weight_noise(self, prototypes):
        if len(prototypes) <= 1:
            self.weight_noise = torch.zeros(len(self.task_adapters), requires_grad=True)
        else:
            self.weight_noise = torch.zeros(len(self.task_adapters), requires_grad=True)
            weight = torch.ones(len(self.task_adapters))
            for i in range(len(prototypes)):
                mu_t = prototypes[-1]
                mu_i = prototypes[i]
                dot_product = torch.dot(mu_t, mu_i)
                norm_t = torch.norm(mu_t)
                norm_i = torch.norm(mu_i)
                s_i = dot_product / (norm_t * norm_i)
                weight[i] = s_i.detach().clone()
            weight = torch.softmax(weight, dim=-1)
            self.weight_noise = weight
            self.weight_noise.requires_grad = True

    def unfreeze_noise(self):
        for param in self.task_adapters[-1].parameters():
            param.requires_grad = True

    def forward(self, hyper_features):
        x1 = self.MLP(hyper_features)
        x_down = hyper_features @ self.w_down

        if len(self.task_adapters) == 0 or self.weight_noise is None:
            return x1 + hyper_features

        adapter_delta = None
        for i, adapter in enumerate(self.task_adapters):
            cur_delta = adapter(x_down)
            if adapter_delta is None:
                adapter_delta = cur_delta * self.weight_noise[i]
            else:
                adapter_delta += cur_delta * self.weight_noise[i]

        adapter_delta = adapter_delta @ self.w_up
        return x1 + adapter_delta + hyper_features

    def forward_new(self, hyper_features):
        x1 = self.MLP(hyper_features)
        x_down = hyper_features @ self.w_down

        if len(self.task_adapters) == 0 or self.weight_noise is None:
            return x1 + hyper_features

        adapter_delta = self.task_adapters[-1](x_down) * self.weight_noise[-1]
        adapter_delta = adapter_delta @ self.w_up
        return x1 + adapter_delta + hyper_features


class VisionTransformer(MiNVisionTransformer):
    def __init__(self, *model_args, args=None, **kwargs):
        super().__init__(*model_args, args=args, **kwargs)

        hidden_dim = args["hidden_dim"]
        device = args.get("device", [None])[0]
        self.noise_maker = nn.Sequential(*[
            PiResidualAdapter(self.embed_dim, self.embed_dim, hidden_dim, device)
            for _ in range(len(self.blocks))
        ])
