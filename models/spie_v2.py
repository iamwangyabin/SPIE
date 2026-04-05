import logging

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

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

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes - self._known_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._cur_task > 0:
            backbone = self._backbone_module()
            backbone.reset_task_modules()

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        if len(self._multiple_gpus) > 1:
            self._network.backbone = nn.DataParallel(self._network.backbone, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network.backbone = self._backbone_module()

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
