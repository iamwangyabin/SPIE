import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm

from models.spie_v2 import Learner as SPiEV2Learner
from models.tuna import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy


class Learner(SPiEV2Learner):
    """SPiE v6 learner with stable shared LoRA and isolated expert LoRA."""

    def __init__(self, args):
        super().__init__(args)
        self.use_orth = False
        self.shared_ssl_lambda = float(args.get("shared_ssl_lambda", 1.0))
        self.cassle_lambda = float(args.get("cassle_lambda", 1.0))
        self.share_lora_lr = float(args.get("share_lora_lr", self.init_lr * args.get("share_lora_lr_scale", 0.1)))
        self.share_lora_weight_decay = float(args.get("share_lora_weight_decay", self.weight_decay))
        self.shared_grad_clip = float(args.get("shared_grad_clip", 1.0))
        self.ssl_flip_p = float(args.get("ssl_flip_p", 0.5))
        self.ssl_noise_std = float(args.get("ssl_noise_std", 0.0))
        self._shared_teacher = None

        for name, param in self._network.backbone.named_parameters():
            param.requires_grad = (
                "cur_adapter" in name
                or "cur_expert_tokens" in name
                or "cur_shared_adapter" in name
                or "cassle_predictor" in name
                or "head" in name
            )

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s SPiE v6 total backbone parameters.", f"{total_params:,}")
        logging.info("%s SPiE v6 trainable backbone parameters.", f"{total_trainable_params:,}")

    def _should_reset_task_modules(self):
        return self._cur_task >= 0

    def get_optimizer(self, model):
        expert_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and ("cur_adapter" in name or "cur_expert_tokens" in name)
        ]
        shared_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and ("cur_shared_adapter" in name or "cassle_predictor" in name)
        ]

        network_params = []
        if expert_params:
            network_params.append(
                {
                    "params": expert_params,
                    "lr": self.init_lr,
                    "weight_decay": self.weight_decay,
                }
            )
        if shared_params:
            network_params.append(
                {
                    "params": shared_params,
                    "lr": self.share_lora_lr,
                    "weight_decay": self.share_lora_weight_decay,
                }
            )
        network_params.append(
            {
                "params": self._network.fc.parameters(),
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            }
        )

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(network_params, momentum=0.9)
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(network_params)
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(network_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")

        return optimizer

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        self._prepare_shared_teacher()
        optimizer = self.get_optimizer(backbone)
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, None, optimizer, scheduler)
        self._shared_teacher = None
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

    def _prepare_shared_teacher(self):
        if self._cur_task <= 0 or self.cassle_lambda <= 0.0:
            self._shared_teacher = None
            return

        teacher = copy.deepcopy(self._backbone_module()).to(self._device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        self._shared_teacher = teacher

    def _shared_trainable_params(self):
        backbone = self._backbone_module()
        params = list(backbone.cur_shared_adapter.parameters())
        params.extend(backbone.cassle_predictor.parameters())
        return [param for param in params if param.requires_grad and param.grad is not None]

    def _set_shared_lora_requires_grad(self, requires_grad):
        backbone = self._backbone_module()
        backbone.cur_shared_adapter.requires_grad_(requires_grad)

    def _ssl_view(self, inputs):
        view = inputs.clone()
        if self.ssl_flip_p > 0.0 and inputs.shape[-1] > 1:
            flip_mask = torch.rand(inputs.shape[0], device=inputs.device) < self.ssl_flip_p
            if flip_mask.any():
                view[flip_mask] = torch.flip(view[flip_mask], dims=[-1])
        if self.ssl_noise_std > 0.0:
            view = view + torch.randn_like(view) * self.ssl_noise_std
        return view

    @staticmethod
    def _negative_cosine_loss(prediction, target):
        prediction = F.normalize(prediction, p=2, dim=1)
        target = F.normalize(target.detach(), p=2, dim=1)
        return 2.0 - 2.0 * (prediction * target).sum(dim=1).mean()

    def _shared_ssl_and_cassle_losses(self, inputs):
        zero = inputs.new_tensor(0.0)
        if self.shared_ssl_lambda <= 0.0 and (self._shared_teacher is None or self.cassle_lambda <= 0.0):
            return zero, zero

        backbone = self._backbone_module()
        view_a = inputs
        view_b = self._ssl_view(inputs)
        cls_a = self._network.backbone(view_a, adapter_id=-1, train=True)["cls_features"]
        cls_b = self._network.backbone(view_b, adapter_id=-1, train=True)["cls_features"]
        pred_a = backbone.cassle_predictor(cls_a)
        pred_b = backbone.cassle_predictor(cls_b)

        ssl_loss = zero
        if self.shared_ssl_lambda > 0.0:
            ssl_loss = 0.5 * (
                self._negative_cosine_loss(pred_a, cls_b)
                + self._negative_cosine_loss(pred_b, cls_a)
            )

        cassle_loss = zero
        if self._shared_teacher is not None and self.cassle_lambda > 0.0:
            with torch.no_grad():
                teacher_a = self._shared_teacher(view_a, adapter_id=-1, train=False)["cls_features"]
                teacher_b = self._shared_teacher(view_b, adapter_id=-1, train=False)["cls_features"]
            cassle_loss = 0.5 * (
                self._negative_cosine_loss(pred_a, teacher_a)
                + self._negative_cosine_loss(pred_b, teacher_b)
            )
        return ssl_loss, cassle_loss

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        del test_loader
        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            if self._shared_teacher is not None:
                self._shared_teacher.eval()

            losses = 0.0
            cls_losses = 0.0
            ssl_losses = 0.0
            cassle_losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                self._set_shared_lora_requires_grad(False)
                try:
                    res = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)
                finally:
                    self._set_shared_lora_requires_grad(True)
                features = res["features"]
                logits = self._network.fc(features)["logits"]

                cls_loss = loss_cos(logits[:, self._known_classes :], targets - self._known_classes)
                ssl_loss, cassle_loss = self._shared_ssl_and_cassle_losses(inputs)
                loss = cls_loss + self.shared_ssl_lambda * ssl_loss + self.cassle_lambda * cassle_loss

                optimizer.zero_grad()
                loss.backward()
                shared_params = self._shared_trainable_params()
                if self.shared_grad_clip > 0.0 and shared_params:
                    nn.utils.clip_grad_norm_(shared_params, max_norm=self.shared_grad_clip)
                optimizer.step()

                losses += loss.item()
                cls_losses += cls_loss.item()
                ssl_losses += ssl_loss.item()
                cassle_losses += cassle_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            avg_cls_loss = cls_losses / len(train_loader)
            avg_ssl_loss = ssl_losses / len(train_loader)
            avg_cassle_loss = cassle_losses / len(train_loader)

            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Cls {:.3f}, SSL {:.3f}, CaSSLe {:.3f}, Train_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                avg_loss,
                avg_cls_loss,
                avg_ssl_loss,
                avg_cassle_loss,
                train_acc,
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.args["tuned_epoch"],
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                cls_loss=float(avg_cls_loss),
                shared_ssl_loss=float(avg_ssl_loss),
                cassle_loss=float(avg_cassle_loss),
                share_lora_lr=float(self.share_lora_lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def orth_loss(self, features):
        del features
        return 0.0
