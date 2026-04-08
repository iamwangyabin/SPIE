import copy
import logging

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from models.spie_v6 import Learner as SPiEV6Learner
from models.tuna import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy


class Learner(SPiEV6Learner):
    """SPiE v7 learner with task0-only shared-LoRA adaptation and isolated expert training."""
    _spie_version_name = "SPiE v7"

    def __init__(self, args):
        super().__init__(args)
        self.task0_shared_epochs = int(args.get("task0_shared_epochs", args["tuned_epoch"]))
        self.task0_expert_epochs = int(args.get("task0_expert_epochs", args["tuned_epoch"]))
        self.task0_shared_lr = float(args.get("task0_shared_lr", self.init_lr * args.get("task0_shared_lr_scale", 1.0)))
        self.task0_expert_lr = float(args.get("task0_expert_lr", self.init_lr))
        self.task0_copy_shared_to_expert = bool(args.get("task0_copy_shared_to_expert", True))
        self.incremental_expert_epochs = int(args.get("incremental_expert_epochs", args["tuned_epoch"]))
        self.incremental_expert_lr = float(
            args.get("incremental_expert_lr", self.init_lr * args.get("incremental_expert_lr_scale", 1.0))
        )

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s %s total backbone parameters.", f"{total_params:,}", self._spie_version_name)
        logging.info("%s %s trainable backbone parameters.", f"{total_trainable_params:,}", self._spie_version_name)

    def _make_optimizer(self, network_params):
        if self.args["optimizer"] == "sgd":
            return optim.SGD(network_params, momentum=0.9)
        if self.args["optimizer"] == "adam":
            return optim.Adam(network_params)
        if self.args["optimizer"] == "adamw":
            return optim.AdamW(network_params)
        raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")

    def _get_scheduler_for_epochs(self, optimizer, epochs):
        if self.args["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=self.min_lr)
        if self.args["scheduler"] == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        if self.args["scheduler"] == "constant":
            return None
        raise ValueError(f"Unsupported scheduler: {self.args['scheduler']}")

    def _set_current_expert_requires_grad(self, requires_grad):
        backbone = self._backbone_module()
        backbone.cur_adapter.requires_grad_(requires_grad)
        backbone.cur_expert_tokens.requires_grad = requires_grad

    def _freeze_shared_domain_adapter(self):
        self._set_shared_lora_requires_grad(False)

    def _copy_shared_lora_to_current_expert(self):
        backbone = self._backbone_module()
        backbone.cur_adapter.load_state_dict(backbone.cur_shared_adapter.state_dict())

    def _task0_shared_optimizer(self):
        backbone = self._backbone_module()
        network_params = [
            {
                "params": [p for p in backbone.cur_shared_adapter.parameters() if p.requires_grad],
                "lr": self.task0_shared_lr,
                "weight_decay": self.share_lora_weight_decay,
            },
            {
                "params": self._network.fc.parameters(),
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    def _task0_expert_optimizer(self):
        backbone = self._backbone_module()
        expert_params = [
            p
            for name, p in backbone.named_parameters()
            if p.requires_grad and ("cur_adapter" in name or "cur_expert_tokens" in name)
        ]
        network_params = [
            {
                "params": expert_params,
                "lr": self.task0_expert_lr,
                "weight_decay": self.weight_decay,
            },
            {
                "params": self._network.fc.parameters(),
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        if self._cur_task == 0:
            self._train_task0_shared_lora(train_loader)
            if self.task0_copy_shared_to_expert and self.task0_shared_epochs > 0:
                self._copy_shared_lora_to_current_expert()
            self._train_task0_expert(train_loader)
        else:
            self._train_incremental_expert(train_loader)

        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

    def _train_task0_shared_lora(self, train_loader):
        if self.task0_shared_epochs <= 0:
            return
        fc_state = copy.deepcopy(self._network.fc.state_dict())
        self._set_shared_lora_requires_grad(True)
        self._set_current_expert_requires_grad(False)
        optimizer = self._task0_shared_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.task0_shared_epochs)
        try:
            self._run_task0_phase(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.task0_shared_epochs,
                adapter_id=-1,
                stage="task0_shared_lora",
            )
        finally:
            self._network.fc.load_state_dict(fc_state)

    def _train_task0_expert(self, train_loader):
        if self.task0_expert_epochs <= 0:
            return
        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(True)
        optimizer = self._task0_expert_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.task0_expert_epochs)
        try:
            self._run_task0_phase(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.task0_expert_epochs,
                adapter_id=self._cur_task,
                stage="task0_expert",
            )
        finally:
            self._set_shared_lora_requires_grad(False)

    def _incremental_expert_optimizer(self):
        backbone = self._backbone_module()
        expert_params = [
            p
            for name, p in backbone.named_parameters()
            if p.requires_grad and ("cur_adapter" in name or "cur_expert_tokens" in name)
        ]
        network_params = [
            {
                "params": expert_params,
                "lr": self.incremental_expert_lr,
                "weight_decay": self.weight_decay,
            },
            {
                "params": self._network.fc.parameters(),
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    def _train_incremental_expert(self, train_loader):
        if self.incremental_expert_epochs <= 0:
            return
        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        optimizer = self._incremental_expert_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.incremental_expert_epochs)
        self._run_incremental_expert_phase(train_loader, optimizer, scheduler)

    def _run_task0_phase(self, train_loader, optimizer, scheduler, epochs, adapter_id, stage):
        if epochs <= 0:
            return

        prog_bar = tqdm(range(epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features = self._network.backbone(inputs, adapter_id=adapter_id, train=True)["features"]
                logits = self._network.fc(features)["logits"]

                loss = loss_cos(logits[:, self._known_classes :], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                stage,
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
            )
            self._record_extra_stage_epoch(
                stage=stage,
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _run_incremental_expert_phase(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.incremental_expert_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._freeze_shared_domain_adapter()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["features"]
                logits = self._network.fc(features)["logits"]

                loss = loss_cos(logits[:, self._known_classes :], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, incremental_expert, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.incremental_expert_epochs,
                avg_loss,
                train_acc,
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.incremental_expert_epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                stage="incremental_expert",
                incremental_expert_lr=float(self.incremental_expert_lr),
                shared_lora_frozen=True,
            )
            prog_bar.set_description(info)

        logging.info(info)
