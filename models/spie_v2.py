import logging

from torch import nn, optim

from models.base import BaseLearner
from models.tunamax import Learner as TunaMaxLearner
from utils.inc_net import TUNANet


class Learner(TunaMaxLearner):
    """TunaMax learner with expert-token backbone."""

    def __init__(self, args):
        BaseLearner.__init__(self, args)

        self._network = TUNANet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.use_orth = args["use_orth"]
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.args["tuned_epoch"] = args["tuned_epoch"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]

        for name, param in self._network.backbone.named_parameters():
            if "adapter" in name or "head" in name or "expert_token" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f"{total_params:,} model total parameters.")
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f"{total_trainable_params:,} model training parameters.")

        if total_params != total_trainable_params:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    logging.info("{}: {}".format(name, param.numel()))

    def get_optimizer(self, model):
        base_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and ("cur_adapter" in name or "cur_expert_token" in name)
        ]
        base_params = {
            "params": base_params,
            "lr": self.init_lr,
            "weight_decay": self.weight_decay,
        }
        base_fc_params = {
            "params": self._network.fc.parameters(),
            "lr": self.init_lr,
            "weight_decay": self.weight_decay,
        }
        network_params = [base_params, base_fc_params]

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(network_params, momentum=0.9)
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(network_params)
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(network_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")

        return optimizer
