import torch
from torch import nn
from torch.nn import functional as F

from backbone.linears import SimpleLinear
from backbone.vit_min import VisionTransformer
from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()
import timm


def _resolve_backbone_name(name):
    lower_name = name.lower()
    if lower_name in {
        "pretrained_vit_b16_224_in21k_min",
        "vit_base_patch16_224_in21k_min",
    }:
        return "vit_base_patch16_224_in21k"
    if lower_name in {
        "pretrained_vit_b16_224_min",
        "vit_base_patch16_224_min",
    }:
        return "vit_base_patch16_224"
    raise NotImplementedError("Unknown MiN backbone type {}".format(name))


def get_min_backbone(args):
    base_model_name = _resolve_backbone_name(args["backbone_type"])
    pretrained = bool(args.get("pretrained", True))
    model_f = timm.create_model(base_model_name, pretrained=pretrained, num_classes=0)
    model = VisionTransformer(num_classes=0, weight_init="skip", args=args)
    model.load_state_dict(model_f.state_dict(), strict=False)
    model.out_dim = getattr(model_f, "num_features", 768)
    model.layer_num = len(model.blocks)
    return model


class RandomBuffer(nn.Module):
    def __init__(self, in_features, buffer_size, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = buffer_size
        self.register_buffer(
            "weight",
            torch.empty((self.in_features, self.out_features), device=device, dtype=torch.double),
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = x.to(device=self.weight.device, dtype=self.weight.dtype)
        return F.relu(x @ self.weight)


class MiNNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args["device"][0]
        self.backbone = get_min_backbone(args)
        self.gamma = float(args.get("gamma", 500))
        self.buffer_size = int(args.get("buffer_size", 16384))
        self.feature_dim = self.backbone.out_dim
        self.task_prototypes = []

        self.buffer = RandomBuffer(
            in_features=self.feature_dim,
            buffer_size=self.buffer_size,
            device=self.device,
        )

        self.register_buffer(
            "weight",
            torch.zeros((self.buffer_size, 0), device=self.device, dtype=torch.double),
        )
        self.register_buffer(
            "R",
            torch.eye(self.buffer_size, device=self.device, dtype=torch.double) / self.gamma,
        )

        self.normal_fc = None
        self.cur_task = -1
        self.known_class = 0

    @property
    def in_features(self):
        return self.weight.shape[0]

    @property
    def out_features(self):
        return self.weight.shape[1]

    def extract_vector(self, x):
        return self.backbone(x)

    def forward_fc(self, features):
        features = features.to(self.weight)
        return features @ self.weight

    def update_fc(self, nb_classes):
        self.cur_task += 1
        self.known_class += nb_classes
        if self.cur_task > 0:
            fc = SimpleLinear(self.buffer_size, self.known_class, bias=False)
        else:
            fc = SimpleLinear(self.buffer_size, nb_classes, bias=True)

        if self.normal_fc is None:
            self.normal_fc = fc.to(self.device)
        else:
            nn.init.constant_(fc.weight, 0.0)
            self.normal_fc = fc.to(self.device)

    @torch.no_grad()
    def fit(self, x, y):
        x = self.buffer(self.backbone(x))
        x = x.to(self.weight)
        y = y.to(self.weight)

        num_targets = y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size), device=self.weight.device, dtype=self.weight.dtype)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((y.shape[0], increment_size), device=y.device, dtype=y.dtype)
            y = torch.cat((y, tail), dim=1)

        k = torch.inverse(torch.eye(x.shape[0], device=x.device, dtype=x.dtype) + x @ self.R @ x.T)
        self.R -= self.R @ x.T @ k @ x @ self.R
        self.weight += self.R @ x.T @ (y - x @ self.weight)

    def forward(self, x, new_forward=False):
        hyper_features = self.backbone(x, new_forward=new_forward)
        logits = self.forward_fc(self.buffer(hyper_features))
        return {"logits": logits}

    def forward_normal_fc(self, x, new_forward=False):
        hyper_features = self.backbone(x, new_forward=new_forward)
        logits = self.normal_fc(self.buffer(hyper_features).to(torch.float32))["logits"]
        return {"logits": logits}

    def update_task_prototype(self, prototype):
        self.task_prototypes[-1] = prototype

    def extend_task_prototype(self, prototype):
        self.task_prototypes.append(prototype)

    def update_noise(self):
        for block_id in range(self.backbone.layer_num):
            self.backbone.noise_maker[block_id].update_noise()
            self.backbone.noise_maker[block_id].init_weight_noise(self.task_prototypes)

    def unfreeze_noise(self):
        for block_id in range(self.backbone.layer_num):
            self.backbone.noise_maker[block_id].unfreeze_noise()

    def init_unfreeze(self):
        for block_id in range(self.backbone.layer_num):
            for param in self.backbone.noise_maker[block_id].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[block_id].norm1.parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[block_id].norm2.parameters():
                param.requires_grad = True
        for param in self.backbone.norm.parameters():
            param.requires_grad = True
