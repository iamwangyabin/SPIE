import logging

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.min_net import MiNNet
from utils.toolkit import tensor2numpy


def _get_optimizer(optimizer_type, params, lr, weight_decay):
    if optimizer_type == "sgd":
        return optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if optimizer_type == "adam":
        return optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    if optimizer_type == "adamw":
        return optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    raise ValueError("Unknown optimizer type {}".format(optimizer_type))


def _get_scheduler(scheduler_type, optimizer, epochs):
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    if scheduler_type == "step":
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[70, 100], gamma=0.1)
    if scheduler_type == "constant":
        return None
    raise ValueError("Unknown scheduler type {}".format(scheduler_type))


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MiNNet(args)
        self.args = args
        self.num_workers = int(args.get("num_workers", 4))

        self.init_epochs = int(args.get("init_epochs", 10))
        self.init_lr = float(args.get("init_lr", 1e-3))
        self.init_weight_decay = float(args.get("init_weight_decay", 5e-4))
        self.init_batch_size = int(args.get("init_batch_size", 128))

        self.epochs = int(args.get("epochs", 10))
        self.lr = float(args.get("lr", self.init_lr))
        self.weight_decay = float(args.get("weight_decay", self.init_weight_decay))
        self.batch_size = int(args.get("batch_size", self.init_batch_size))

        self.buffer_batch = int(args.get("buffer_batch", 1000))
        self.fit_epochs = int(args.get("fit_epochs", 3))
        self.optimizer_type = args.get("optimizer_type", "sgd").lower()
        self.scheduler_type = args.get("scheduler_type", "step").lower()

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info("Learning on %d-%d", self._known_classes, self._total_classes)

        current_indices = np.arange(self._known_classes, self._total_classes)
        seen_indices = np.arange(0, self._total_classes)

        train_dataset = data_manager.get_dataset(current_indices, source="train", mode="train")
        plain_train_dataset = data_manager.get_dataset(current_indices, source="train", mode="test")
        test_dataset = data_manager.get_dataset(seen_indices, source="test", mode="test")

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.buffer_batch,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if self._cur_task == 0:
            self._network.update_fc(data_manager.get_task_size(self._cur_task))
            self._network.update_noise()
            run_loader = DataLoader(
                train_dataset,
                batch_size=self.init_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            prototype = self._get_task_prototype(run_loader)
            self._network.extend_task_prototype(prototype)
            self._run_noise_learning(run_loader)
            self._network.update_task_prototype(self._get_task_prototype(run_loader))

            fit_loader = DataLoader(
                train_dataset,
                batch_size=self.buffer_batch,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self._fit_fc(fit_loader)
        else:
            fit_loader = DataLoader(
                train_dataset,
                batch_size=self.buffer_batch,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self._fit_fc(fit_loader)

            self._network.update_fc(data_manager.get_task_size(self._cur_task))
            run_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self._network.update_noise()
            self._network.extend_task_prototype(self._get_task_prototype(run_loader))
            self._run_noise_learning(run_loader)
            self._network.update_task_prototype(self._get_task_prototype(run_loader))

        plain_train_loader = DataLoader(
            plain_train_dataset,
            batch_size=self.buffer_batch,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self._re_fit(plain_train_loader)

    def _fit_fc(self, train_loader):
        self._network.eval()
        self._network.to(self._device)

        for epoch in range(self.fit_epochs):
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = F.one_hot(targets.to(self._device), num_classes=self._total_classes)
                self._network.fit(inputs, targets)

            self._record_extra_stage_epoch(
                stage="fit_fc",
                epoch=epoch + 1,
                total_epochs=self.fit_epochs,
            )
            logging.info("Task %d --> Update analytical classifier (%d/%d)", self._cur_task, epoch + 1, self.fit_epochs)

    def _re_fit(self, train_loader):
        self._network.eval()
        self._network.to(self._device)

        total_batches = len(train_loader)
        for batch_idx, (_, inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(self._device)
            targets = F.one_hot(targets.to(self._device), num_classes=self._total_classes)
            self._network.fit(inputs, targets)

            logging.info(
                "Task %d --> Reupdate analytical classifier (%d/%d)",
                self._cur_task,
                batch_idx,
                total_batches,
            )

        self._record_extra_stage_epoch(
            stage="re_fit",
            epoch=1,
            total_epochs=1,
            batches=total_batches,
        )

    def _run_noise_learning(self, train_loader):
        if self._cur_task == 0:
            epochs = self.init_epochs
            lr = self.init_lr
            weight_decay = self.init_weight_decay
        else:
            epochs = self.epochs
            lr = self.lr
            weight_decay = self.weight_decay

        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.normal_fc.parameters():
            param.requires_grad = True

        if self._cur_task == 0:
            self._network.init_unfreeze()
        else:
            self._network.unfreeze_noise()

        params = [param for param in self._network.parameters() if param.requires_grad]
        optimizer = _get_optimizer(self.optimizer_type, params, lr, weight_decay)
        scheduler = _get_scheduler(self.scheduler_type, optimizer, epochs)

        self._network.train()
        self._network.to(self._device)
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                if self._cur_task > 0:
                    with torch.no_grad():
                        analytical_logits = self._network(inputs, new_forward=False)["logits"]
                    normal_logits = self._network.forward_normal_fc(inputs, new_forward=False)["logits"]
                    logits = normal_logits + analytical_logits
                else:
                    logits = self._network.forward_normal_fc(inputs, new_forward=False)["logits"]

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / max(len(train_loader), 1)
            lr_now = optimizer.param_groups[0]["lr"]

            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr_now),
            )

            info = (
                "Task {} --> Learning Beneficial Noise!: Epoch {}/{} => Loss {:.3f}, train_accy {:.2f}"
            ).format(self._cur_task, epoch + 1, epochs, avg_loss, train_acc)
            logging.info(info)
            prog_bar.set_description(info)

    def _get_task_prototype(self, train_loader):
        self._network.eval()
        self._network.to(self._device)
        features = []

        with torch.no_grad():
            for _, inputs, _ in train_loader:
                inputs = inputs.to(self._device)
                features.append(self._network.extract_vector(inputs))

        return torch.mean(torch.cat(features, dim=0), dim=0)
