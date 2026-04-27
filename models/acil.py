import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.inc_net import get_backbone


class RandomFeatureBuffer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight",
            torch.empty(in_features, out_features, device=device, dtype=torch.float32),
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = x.to(device=self.weight.device, dtype=self.weight.dtype)
        return F.relu(x @ self.weight)


class ACILNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args["device"][0]
        self.backbone = get_backbone(args, pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_dim = self.backbone.out_dim
        self.buffer_size = int(args.get("buffer_size", 2048))
        self.gamma = float(args.get("gamma", 1.0))
        self.use_input_norm = bool(args.get("use_input_norm", False))

        self.buffer = RandomFeatureBuffer(self.feature_dim, self.buffer_size, self.device)
        self.register_buffer(
            "weight",
            torch.zeros(self.buffer_size, 0, device=self.device, dtype=torch.float32),
        )
        self.register_buffer(
            "R",
            torch.eye(self.buffer_size, device=self.device, dtype=torch.float32) / self.gamma,
        )

    @property
    def out_features(self):
        return self.weight.shape[1]

    def extract_vector(self, x):
        return self.backbone(x)

    def _project(self, x):
        features = self.extract_vector(x)
        if self.use_input_norm:
            features = F.normalize(features, p=2, dim=1)
        return self.buffer(features)

    def update_fc(self, nb_classes):
        if nb_classes <= self.out_features:
            return
        tail = torch.zeros(
            self.buffer_size,
            nb_classes - self.out_features,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.weight = torch.cat((self.weight, tail), dim=1)

    @torch.no_grad()
    def fit(self, x, y):
        x = self._project(x).to(self.weight)
        y = y.to(self.weight)

        if y.shape[1] > self.out_features:
            self.update_fc(y.shape[1])
        elif y.shape[1] < self.out_features:
            tail = torch.zeros(
                y.shape[0],
                self.out_features - y.shape[1],
                device=y.device,
                dtype=y.dtype,
            )
            y = torch.cat((y, tail), dim=1)

        eye = torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        middle = torch.linalg.solve(eye + x @ self.R @ x.T, x @ self.R)
        self.R -= self.R @ x.T @ middle
        self.weight += self.R @ x.T @ (y - x @ self.weight)

    def forward(self, x):
        logits = self._project(x) @ self.weight
        return {"logits": logits}


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = ACILNet(args)
        self.batch_size = int(args.get("batch_size", 128))
        self.fit_batch_size = int(args.get("fit_batch_size", self.batch_size))
        self.num_workers = int(args.get("num_workers", 4))

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on %d-%d", self._known_classes, self._total_classes)

        current_indices = np.arange(self._known_classes, self._total_classes)
        seen_indices = np.arange(0, self._total_classes)

        train_dataset = data_manager.get_dataset(current_indices, source="train", mode="test")
        test_dataset = data_manager.get_dataset(seen_indices, source="test", mode="test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.fit_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self._network.eval()
        self._network.to(self._device)
        total_batches = len(train_loader)
        for batch_idx, (_, inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(self._device)
            targets = F.one_hot(targets.to(self._device), num_classes=self._total_classes)
            self._network.fit(inputs, targets)
            logging.info(
                "Task %d --> Fit analytical classifier (%d/%d)",
                self._cur_task,
                batch_idx,
                total_batches,
            )

        self._record_extra_stage_epoch(
            stage="fit_acil",
            epoch=1,
            total_epochs=1,
            batches=total_batches,
        )
