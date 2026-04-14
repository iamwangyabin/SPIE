import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from models.tuna import AngularPenaltySMLoss
from utils.inc_net import TUNANet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = TUNANet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = float(args["init_lr"])
        self.weight_decay = float(args.get("weight_decay", 5e-4) or 5e-4)
        self.min_lr = float(args.get("min_lr", 0.0) or 0.0)
        self.args = args
        self.loss_cos = AngularPenaltySMLoss(
            loss_type=args.get("loss_type", "cosface"),
            eps=1e-7,
            s=args.get("scale", 20.0),
            m=args.get("m", 0.0),
        )

        self.prompt_penalty_weight = float(args.get("prompt_penalty_weight", 0.5))
        self.aux_weight = float(args.get("aux_weight", 0.1))
        self.aux_tau = float(args.get("aux_tau", 0.01))
        self.enable_greedy_init = bool(args.get("greedy_init", True))
        self._old_backbone = None

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s KA-Prompt backbone parameters.", f"{total_params:,}")
        logging.info("%s KA-Prompt trainable backbone parameters.", f"{trainable_params:,}")

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_backbone = copy.deepcopy(self._network.backbone).to(self._device)
        self._old_backbone.eval()
        self._old_backbone.requires_grad_(False)

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._network.update_fc(self._total_classes - self._known_classes)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

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
            num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        greedy_loader = self.train_loader if self.enable_greedy_init else None
        self._network.backbone.begin_task(self._cur_task, greedy_loader)
        self._train(self.train_loader, self.test_loader)

    def _make_optimizer(self):
        prompt_params = [p for p in self._network.backbone.prompt_parameters() if p.requires_grad]
        network_params = [
            {"params": prompt_params, "lr": self.init_lr, "weight_decay": self.weight_decay},
            {"params": self._network.fc.parameters(), "lr": self.init_lr, "weight_decay": self.weight_decay},
        ]

        optimizer_name = self.args.get("optimizer", "adamw")
        if optimizer_name == "sgd":
            return optim.SGD(network_params, momentum=0.9)
        if optimizer_name == "adam":
            return optim.Adam(network_params)
        if optimizer_name == "adamw":
            return optim.AdamW(network_params)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _make_scheduler(self, optimizer):
        scheduler_name = self.args.get("scheduler", "cosine")
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["tuned_epoch"],
                eta_min=self.min_lr,
            )
        if scheduler_name == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        if scheduler_name == "constant":
            return None
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _forward_current(self, inputs):
        outputs = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)
        logits = self._network.fc(outputs["features"])["logits"]
        return outputs, logits

    def _forward_old_mix(self, inputs, keys):
        if self._old_backbone is None or self._cur_task <= 0:
            return None

        old_outputs = self._network.backbone.forward_old_prompt_mix(
            inputs,
            task_id=self._cur_task,
            new_keys=keys,
            old_backbone=self._old_backbone,
            tau=self.aux_tau,
        )
        logits = self._network.fc(old_outputs["features"])["logits"]
        return logits

    def _classification_loss(self, logits, targets):
        current_logits = logits[:, self._known_classes :]
        current_targets = targets - self._known_classes
        return self.loss_cos(current_logits, current_targets)

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        self._network.fc.to(self._device)
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)

        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs, logits = self._forward_current(inputs)
                loss = self._classification_loss(logits, targets)

                distance = outputs.get("distance")
                if distance is not None:
                    loss = loss + self.prompt_penalty_weight * distance.sum(dim=1).mean()

                if self.aux_weight > 0 and self._cur_task > 0:
                    mixed_logits = self._forward_old_mix(inputs, outputs["keys"])
                    loss = loss + self.aux_weight * self._classification_loss(mixed_logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = torch.max(logits[:, : self._total_classes], dim=1)[1]
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                avg_loss,
                train_acc,
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.args["tuned_epoch"],
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.backbone.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network.backbone(inputs, adapter_id=self._cur_task, train=False)["features"]
                outputs = self._network.fc(features)["logits"][:, : self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _extract_vectors(self, loader):
        self._network.backbone.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, inputs, _targets in loader:
                inputs = inputs.to(self._device)
                features = self._network.backbone(inputs, adapter_id=self._cur_task, train=False)["features"]
                vectors.append(tensor2numpy(features))
                targets.append(_targets.numpy())
        return np.concatenate(vectors), np.concatenate(targets)
