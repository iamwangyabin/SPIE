import copy

import timm
import torch
from torch import nn

from backbone.linears import AC_Linear, CosineLinear
from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()


def get_moal_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    if "_moal" not in name:
        raise NotImplementedError("MoAL expects a backbone type ending with '_moal'.")

    from backbone import vit_tuna
    from easydict import EasyDict

    tuning_config = EasyDict(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args["ffn_num"],
        d_model=768,
        vpt_on=False,
        vpt_num=0,
    )
    setattr(tuning_config, "_device", args["device"][0])

    if name == "vit_base_patch16_224_moal":
        model = vit_tuna.vit_base_patch16_224_adapter(
            num_classes=0,
            global_pool=False,
            drop_path_rate=0.0,
            tuning_config=tuning_config,
        )
    elif name == "vit_base_patch16_224_in21k_moal":
        model = vit_tuna.vit_base_patch16_224_in21k_adapter(
            num_classes=0,
            global_pool=False,
            drop_path_rate=0.0,
            tuning_config=tuning_config,
        )
    else:
        raise NotImplementedError("Unknown MoAL backbone type {}".format(name))

    model.out_dim = 768
    return model.eval()


def get_plain_backbone(backbone_type):
    if backbone_type == "vit_base_patch16_224_moal":
        plain_name = "vit_base_patch16_224"
    elif backbone_type == "vit_base_patch16_224_in21k_moal":
        plain_name = "vit_base_patch16_224_in21k"
    else:
        raise NotImplementedError("Unknown MoAL backbone type {}".format(backbone_type))

    model = timm.create_model(plain_name, pretrained=True, num_classes=0)
    model.out_dim = 768
    return model.eval()


class SimpleVitNetAL(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_moal_backbone(args, pretrained)
        self.fc = None
        self.ac_model = None
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes, hidden):
        ac_model = self.generate_ac(self.feature_dim, hidden, nb_classes).to(self._device)
        if self.ac_model is not None:
            prev_out = self.ac_model.out_features
            ac_model.fc[0].weight = nn.Parameter(self.ac_model.fc[0].weight.data.clone().float())
            weight = self.ac_model.fc[-1].weight.data.clone()
            extra = torch.zeros(nb_classes - prev_out, hidden, device=self._device)
            ac_model.fc[-1].weight = nn.Parameter(torch.cat([weight, extra], dim=0).float())
        self.ac_model = ac_model

    def generate_ac(self, in_dim, hidden, out_dim):
        return AC_Linear(in_dim, hidden, out_dim)

    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        features = self.backbone(x)
        if self.ac_model is None:
            out = self.fc(features)
            out["train_logits"] = out["logits"]
        else:
            out = self.ac_model(features)
            if self.fc is not None:
                out["train_logits"] = self.fc(features)["logits"]
        out["features"] = features
        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class MoALNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.args = args
        self._device = args["device"][0]
        self.backbones = nn.ModuleList()
        self.fc = None
        self.ac_model = None
        self._feature_dim = 0

    @property
    def feature_dim(self):
        return self._feature_dim

    def generate_fc(self, in_dim, out_dim):
        return CosineLinear(in_dim, out_dim)

    def generate_ac(self, in_dim, hidden, out_dim):
        return AC_Linear(in_dim, hidden, out_dim)

    def update_fc(self, nb_classes, hidden, cosine_fc=False):
        if cosine_fc:
            fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
            if self.fc is not None:
                prev_out = self.fc.out_features
                fc.sigma.data = self.fc.sigma.data.clone()
                extra = torch.zeros(nb_classes - prev_out, self.feature_dim, device=self._device)
                fc.weight = nn.Parameter(torch.cat([self.fc.weight.data.clone(), extra], dim=0))
            self.fc = fc
            return

        ac_model = self.generate_ac(self.feature_dim, hidden, nb_classes).to(self._device)
        if self.ac_model is not None:
            prev_out = self.ac_model.out_features
            ac_model.fc[0].weight = nn.Parameter(self.ac_model.fc[0].weight.data.clone().float())
            extra = torch.zeros(nb_classes - prev_out, hidden, device=self._device)
            ac_model.fc[-1].weight = nn.Parameter(
                torch.cat([self.ac_model.fc[-1].weight.data.clone(), extra], dim=0).float()
            )
        self.ac_model = ac_model

    def construct_dual_branch_network(self, tuned_model):
        self.backbones.append(get_plain_backbone(self.args["backbone_type"]))
        self.backbones.append(tuned_model.backbone)
        self._feature_dim = sum(backbone.out_dim for backbone in self.backbones)
        self.fc = self.generate_fc(self._feature_dim, self.args["init_cls"]).to(self._device)

    def extract_vector(self, x):
        features = [backbone(x) for backbone in self.backbones]
        return torch.cat(features, dim=1)

    def forward(self, x):
        features = self.extract_vector(x)
        out = self.fc(features)
        if self.ac_model is not None:
            out = self.ac_model(features)
            out["train_logits"] = self.fc(features)["logits"]
        else:
            out["train_logits"] = out["logits"]
        out["features"] = features
        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
