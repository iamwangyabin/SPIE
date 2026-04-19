import logging

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import TunaLinear
from models.base import BaseLearner
from utils.inc_net import get_backbone
from utils.toolkit import tensor2numpy

num_workers = 8


class MQMKNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args, pretrained=args.get("pretrained", True))
        self.backbone.out_dim = getattr(self.backbone, "out_dim", 768)
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = TunaLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = MQMKNet(args)
        self.batch_size = args["batch_size"]
        self.init_lr = float(args["init_lr"])
        self.weight_decay = float(args.get("weight_decay", 0.0) or 0.0)
        self.min_lr = float(args.get("min_lr", 0.0) or 0.0)
        self.args = args
        self.criterion = nn.CrossEntropyLoss()

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("MQMK base pretrained backbone: %s", getattr(self._network.backbone, "base_model_name", "unknown"))
        logging.info("%s MQMK backbone parameters.", f"{total_params:,}")
        logging.info("%s MQMK trainable backbone parameters.", f"{trainable_params:,}")

    def after_task(self):
        self._known_classes = self._total_classes

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
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self._network.backbone.begin_task(self._cur_task, self.train_loader)
        self._train(self.train_loader)

    def _make_optimizer(self):
        prompt_params = [p for p in self._network.backbone.prompt_parameters() if p.requires_grad]
        params = [
            {"params": prompt_params, "lr": self.init_lr, "weight_decay": self.weight_decay},
            {"params": self._network.fc.parameters(), "lr": self.init_lr, "weight_decay": self.weight_decay},
        ]

        optimizer_name = self.args.get("optimizer", "adam")
        if optimizer_name == "sgd":
            return optim.SGD(params, momentum=0.9)
        if optimizer_name == "adam":
            return optim.Adam(params)
        if optimizer_name == "adamw":
            return optim.AdamW(params)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _make_scheduler(self, optimizer):
        scheduler_name = self.args.get("scheduler", "constant")
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

    def _build_cls_features(self, inputs):
        if not self.args.get("multi_query", True):
            return None

        feature_list = []
        for task_id in range(self._cur_task + 1):
            query_outputs = self._network.backbone(inputs, adapter_id=task_id, query=True)
            feature_list.append(query_outputs["cls_features"])

        zero_feature = torch.zeros_like(feature_list[0])
        for _ in range(self._cur_task + 1, self.args["nb_tasks"]):
            feature_list.append(zero_feature)

        return torch.stack(feature_list, dim=1)

    def _forward_current(self, inputs, targets=None, train=False):
        cls_features = self._build_cls_features(inputs)
        outputs = self._network.backbone(
            inputs,
            adapter_id=self._cur_task,
            train=train,
            cls_features=cls_features,
            target=targets,
            query=False,
        )
        logits = self._network.fc(outputs["features"])["logits"]
        return outputs, logits

    def _classification_loss(self, logits, targets):
        current_logits = logits[:, self._known_classes :]
        current_targets = targets - self._known_classes
        return self.criterion(current_logits, current_targets)

    def _train(self, train_loader):
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

                outputs, logits = self._forward_current(inputs, targets=targets, train=True)
                loss = self._classification_loss(logits, targets)

                if self.args.get("pull_constraint", True) and outputs.get("reduce_sim") is not None:
                    loss = loss - float(self.args.get("pull_constraint_coeff", 1.0)) * outputs["reduce_sim"]

                optimizer.zero_grad()
                loss.backward()
                if self.args.get("clip_grad", None):
                    torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.args["clip_grad"])
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
                outputs, logits = self._forward_current(inputs, targets=None, train=False)
                outputs = logits[:, : self._total_classes]
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
                outputs, _ = self._forward_current(inputs, targets=None, train=False)
                vectors.append(tensor2numpy(outputs["features"]))
                targets.append(_targets.numpy())
        return np.concatenate(vectors), np.concatenate(targets)
