import copy
import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from models.tuna import AngularPenaltySMLoss
from utils.inc_net import TUNANet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    """Standalone SPiE v13 learner."""

    _spie_version_name = "SPiE v13"

    def __init__(self, args):
        super().__init__(args)

        args["enable_shared_cls_classifier"] = True
        args["enable_expert_calibration"] = True

        self._network = TUNANet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()
        self.shared_cls_mean = dict()
        self.shared_cls_cov = dict()

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.args["tuned_epoch"] = args["tuned_epoch"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]

        self.share_lora_weight_decay = float(args.get("share_lora_weight_decay", self.weight_decay))

        self.task0_shared_epochs = int(args.get("task0_shared_epochs", args["tuned_epoch"]))
        self.task0_expert_epochs = int(args.get("task0_expert_epochs", args["tuned_epoch"]))
        self.task0_shared_lr = float(args.get("task0_shared_lr", self.init_lr * args.get("task0_shared_lr_scale", 1.0)))
        self.task0_expert_lr = float(args.get("task0_expert_lr", self.init_lr))
        self.incremental_expert_epochs = int(args.get("incremental_expert_epochs", args["tuned_epoch"]))
        self.incremental_expert_lr = float(
            args.get("incremental_expert_lr", self.init_lr * args.get("incremental_expert_lr_scale", 1.0))
        )

        self.shared_update_epochs = int(args.get("shared_update_epochs", 3))
        self.shared_update_lr = float(
            args.get("shared_update_lr", self.init_lr * args.get("shared_update_lr_scale", 0.02))
        )
        self.shared_ema_alpha = float(args.get("shared_ema_alpha", 0.05))

        self.expert_calibration_epochs = int(args.get("expert_calibration_epochs", 5))
        self.expert_calibration_lr = float(
            args.get("expert_calibration_lr", self.init_lr * args.get("expert_calibration_lr_scale", 0.1))
        )
        self.expert_calibration_weight_decay = float(args.get("expert_calibration_weight_decay", 0.0))
        self.expert_calibration_routing_temperature = float(args.get("expert_calibration_routing_temperature", 1.0))

        self.shared_cls_epochs = int(args.get("shared_cls_epochs", args["tuned_epoch"]))
        self.shared_cls_lr = float(args.get("shared_cls_lr", self.init_lr))
        self.shared_cls_weight_decay = float(args.get("shared_cls_weight_decay", self.weight_decay))
        self.shared_cls_ca_lr = float(args.get("shared_cls_ca_lr", self.ca_lr))
        self.shared_cls_crct_epochs = int(args.get("shared_cls_crct_epochs", self.crct_epochs))

        if not 0.0 <= self.shared_ema_alpha <= 1.0:
            raise ValueError(f"shared_ema_alpha must be in [0, 1], got {self.shared_ema_alpha}")

        for name, param in self._network.backbone.named_parameters():
            param.requires_grad = (
                "cur_adapter" in name
                or "cur_expert_tokens" in name
                or "cur_shared_adapter" in name
                or "head" in name
            )

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s %s total backbone parameters.", f"{total_params:,}", self._spie_version_name)
        logging.info("%s %s trainable backbone parameters.", f"{total_trainable_params:,}", self._spie_version_name)
        logging.info(
            "SPiE v13 shared update: epochs=%s, lr=%s, ema_alpha=%s.",
            self.shared_update_epochs,
            self.shared_update_lr,
            self.shared_ema_alpha,
        )
        logging.info(
            "SPiE v13 expert calibration: epochs=%s, lr=%s, routing_temperature=%s.",
            self.expert_calibration_epochs,
            self.expert_calibration_lr,
            self.expert_calibration_routing_temperature,
        )
        logging.info(
            "SPiE v13 shared-CLS classifier: epochs=%s, lr=%s, crct_epochs=%s, crct_lr=%s.",
            self.shared_cls_epochs,
            self.shared_cls_lr,
            self.shared_cls_crct_epochs,
            self.shared_cls_ca_lr,
        )

    def _backbone_module(self):
        if isinstance(self._network.backbone, nn.DataParallel):
            return self._network.backbone.module
        return self._network.backbone

    def after_task(self):
        self._known_classes = self._total_classes

    def _should_reset_task_modules(self):
        return self._cur_task >= 0

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for class_idx in range(self._known_classes, self._total_classes):
            self.cls2task[class_idx] = self._cur_task

        self._network.update_fc(self._total_classes - self._known_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._should_reset_task_modules():
            self._backbone_module().reset_task_modules()

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

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        use_backbone_dataparallel = bool(self.args.get("spie_v13_backbone_dataparallel", False))
        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = nn.DataParallel(self._network.backbone, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = self._backbone_module()

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

    def _set_shared_lora_requires_grad(self, requires_grad):
        self._backbone_module().cur_shared_adapter.requires_grad_(requires_grad)

    def _set_current_expert_requires_grad(self, requires_grad):
        backbone = self._backbone_module()
        backbone.cur_adapter.requires_grad_(requires_grad)
        backbone.cur_expert_tokens.requires_grad = requires_grad

    def _freeze_shared_domain_adapter(self):
        self._set_shared_lora_requires_grad(False)

    def _swap_in_temporary_fc(self):
        main_fc = self._network.fc
        temporary_fc = copy.deepcopy(main_fc).to(self._device)
        for param in temporary_fc.parameters():
            param.requires_grad = True
        self._network.fc = temporary_fc
        return main_fc

    def _restore_main_fc(self, main_fc):
        self._network.fc = main_fc
        self._network.fc.to(self._device)

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

    def _shared_delta_optimizer(self):
        backbone = self._backbone_module()
        network_params = [
            {
                "params": [p for p in backbone.cur_shared_adapter.parameters() if p.requires_grad],
                "lr": self.shared_update_lr,
                "weight_decay": self.share_lora_weight_decay,
            },
            {
                "params": self._network.fc.parameters(),
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    @staticmethod
    def _restore_requires_grad(params, flags):
        for param, requires_grad in zip(params, flags):
            param.requires_grad = requires_grad

    def _build_safe_distribution(self, mean, covariance):
        mean = mean.to(self._device, dtype=torch.float32)
        if covariance.ndim == 1:
            covariance = torch.diag(covariance)

        covariance = covariance.to(self._device, dtype=torch.float64)
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.T)

        base_jitter = float(self.args.get("covariance_regularization", 1e-4))
        max_retry_power = int(self.args.get("max_covariance_retry_power", 6))
        eye = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)

        for power in range(max_retry_power + 1):
            jitter = base_jitter * (10 ** power)
            repaired_covariance = covariance + eye * jitter
            _, info = torch.linalg.cholesky_ex(repaired_covariance)
            if torch.all(info == 0):
                scale_tril = torch.linalg.cholesky(repaired_covariance).to(dtype=torch.float32)
                return MultivariateNormal(mean, scale_tril=scale_tril)

        min_eigenvalue = torch.linalg.eigvalsh(covariance).min().item()
        jitter = max(base_jitter, -min_eigenvalue + base_jitter)
        repaired_covariance = covariance + eye * jitter
        scale_tril = torch.linalg.cholesky(repaired_covariance).to(dtype=torch.float32)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        if self._network.fc_shared_cls is not None:
            self._network.fc_shared_cls.to(self._device)

        if self._cur_task == 0:
            self._train_task0_shared_lora(train_loader)
            self._train_task0_expert(train_loader)
        else:
            self._train_incremental_expert(train_loader)
            self._train_shared_delta(train_loader)

        self._train_expert_calibration(train_loader)

        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

        self._train_shared_cls_classifier(train_loader)
        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

    def _train_task0_shared_lora(self, train_loader):
        if self.task0_shared_epochs <= 0:
            return

        main_fc = self._swap_in_temporary_fc()
        try:
            self._set_shared_lora_requires_grad(True)
            self._set_current_expert_requires_grad(False)
            optimizer = self._task0_shared_optimizer()
            scheduler = self._get_scheduler_for_epochs(optimizer, self.task0_shared_epochs)
            self._run_task0_phase(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.task0_shared_epochs,
                adapter_id=-1,
                stage="task0_shared_lora",
            )
        finally:
            self._restore_main_fc(main_fc)

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

    def _train_incremental_expert(self, train_loader):
        if self.incremental_expert_epochs <= 0:
            return

        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        optimizer = self._incremental_expert_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.incremental_expert_epochs)
        self._run_incremental_expert_phase(train_loader, optimizer, scheduler)

    def _train_shared_delta(self, train_loader):
        if self.shared_update_epochs <= 0 or self.shared_ema_alpha <= 0.0:
            return

        backbone = self._backbone_module()
        main_fc = self._swap_in_temporary_fc()
        original_shared_adapter = None
        try:
            original_shared_adapter = backbone.cur_shared_adapter
            shared_work = copy.deepcopy(original_shared_adapter).to(self._device)

            original_shared_adapter.requires_grad_(False)
            self._set_current_expert_requires_grad(False)
            shared_work.requires_grad_(True)
            backbone.cur_shared_adapter = shared_work

            optimizer = self._shared_delta_optimizer()
            scheduler = self._get_scheduler_for_epochs(optimizer, self.shared_update_epochs)
            self._run_task0_phase(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.shared_update_epochs,
                adapter_id=-1,
                stage="shared_delta",
            )
            self._ema_update_shared_adapter(original_shared_adapter, shared_work)
        finally:
            if original_shared_adapter is not None:
                backbone.cur_shared_adapter = original_shared_adapter
            self._restore_main_fc(main_fc)
            self._freeze_shared_domain_adapter()

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

    @torch.no_grad()
    def _ema_update_shared_adapter(self, target_shared_adapter, source_shared_adapter):
        for target_param, source_param in zip(target_shared_adapter.parameters(), source_shared_adapter.parameters()):
            source_data = source_param.data.to(device=target_param.device, dtype=target_param.dtype)
            target_param.data.add_(source_data - target_param.data, alpha=self.shared_ema_alpha)

    def _active_expert_ids(self):
        return list(range(self._cur_task + 1))

    def _expert_calibration_optimizer(self):
        params = []
        for expert_id in self._active_expert_ids():
            params.append(self._network.expert_calibration_log_scale[expert_id])
            params.append(self._network.expert_calibration_bias[expert_id])
        network_params = [
            {
                "params": params,
                "lr": self.expert_calibration_lr,
                "weight_decay": self.expert_calibration_weight_decay,
            }
        ]
        return self._make_optimizer(network_params)

    def _train_expert_calibration(self, train_loader):
        if self.expert_calibration_epochs <= 0:
            return

        active_expert_ids = self._active_expert_ids()
        if not active_expert_ids:
            return

        optimizer = self._expert_calibration_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.expert_calibration_epochs)
        prog_bar = tqdm(range(self.expert_calibration_epochs))

        backbone_params = list(self._network.backbone.parameters())
        fc_params = list(self._network.fc.parameters())
        backbone_grad_flags = [param.requires_grad for param in backbone_params]
        fc_grad_flags = [param.requires_grad for param in fc_params]

        try:
            self._network.backbone.eval()
            self._network.fc.eval()
            for param in backbone_params:
                param.requires_grad = False
            for param in fc_params:
                param.requires_grad = False

            routing_temperature = max(self.expert_calibration_routing_temperature, 1e-8)
            for _, epoch in enumerate(prog_bar):
                losses = 0.0
                correct, total = 0, 0

                for _, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    frozen_logits = []
                    with torch.no_grad():
                        for expert_id in active_expert_ids:
                            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
                            logits = self._network.fc(features)["logits"][:, : self._total_classes]
                            frozen_logits.append(logits.detach())

                    stacked_logits = torch.stack(
                        [
                            self._network.calibrate_expert_logits(logits, expert_id)
                            for expert_id, logits in zip(active_expert_ids, frozen_logits)
                        ],
                        dim=0,
                    )
                    expert_scores = stacked_logits.max(dim=2).values
                    expert_weights = F.softmax(expert_scores / routing_temperature, dim=0)
                    fused_logits = (expert_weights.unsqueeze(-1) * stacked_logits).sum(dim=0)

                    loss = F.cross_entropy(self.args["scale"] * fused_logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses += loss.item()
                    _, preds = torch.max(fused_logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                if scheduler:
                    scheduler.step()

                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = losses / len(train_loader)
                info = "Task {}, expert_calibration, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.expert_calibration_epochs,
                    avg_loss,
                    train_acc,
                )
                self._record_extra_stage_epoch(
                    stage="expert_calibration",
                    epoch=epoch + 1,
                    total_epochs=self.expert_calibration_epochs,
                    loss=float(avg_loss),
                    acc=float(train_acc),
                    lr=float(lr),
                )
                prog_bar.set_description(info)
        finally:
            self._restore_requires_grad(backbone_params, backbone_grad_flags)
            self._restore_requires_grad(fc_params, fc_grad_flags)

        logging.info(info)

    def _shared_cls_optimizer(self):
        network_params = [
            {
                "params": self._network.fc_shared_cls.parameters(),
                "lr": self.shared_cls_lr,
                "weight_decay": self.shared_cls_weight_decay,
            }
        ]
        return self._make_optimizer(network_params)

    def _train_shared_cls_classifier(self, train_loader):
        if self.shared_cls_epochs <= 0 or self._network.fc_shared_cls is None:
            return

        optimizer = self._shared_cls_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.shared_cls_epochs)
        prog_bar = tqdm(range(self.shared_cls_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        self._network.backbone.eval()
        for _, epoch in enumerate(prog_bar):
            self._network.fc_shared_cls.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.no_grad():
                    cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
                logits = self._network.fc_shared_cls(cls_features)["logits"]

                loss = loss_cos(logits[:, self._known_classes :], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits[:, self._known_classes :], dim=1)
                preds = preds + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, shared_cls_classifier, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.shared_cls_epochs,
                avg_loss,
                train_acc,
            )
            self._record_extra_stage_epoch(
                stage="shared_cls_classifier",
                epoch=epoch + 1,
                total_epochs=self.shared_cls_epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _stack_calibrated_expert_logits(self, inputs, active_expert_ids):
        logits_per_expert = []
        for expert_id in active_expert_ids:
            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
            logits = self._network.fc(features)["logits"][:, : self._total_classes]
            logits_per_expert.append(self._network.calibrate_expert_logits(logits, expert_id))
        return torch.stack(logits_per_expert, dim=0)

    def _fuse_calibrated_expert_logits(self, stacked_logits):
        routing_temperature = max(self.expert_calibration_routing_temperature, 1e-8)
        expert_scores = stacked_logits.max(dim=2).values
        expert_weights = F.softmax(expert_scores / routing_temperature, dim=0)
        return (expert_weights.unsqueeze(-1) * stacked_logits).sum(dim=0)

    def _shared_cls_logits(self, inputs):
        if self._network.fc_shared_cls is None:
            return None
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        return self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_expert_ids = self._active_expert_ids()
        use_shared_cls = bool(self.args.get("v9_use_shared_cls_eval", True))
        expert_weight = float(self.args.get("v9_expert_fusion_weight", 1.0))
        shared_weight = float(self.args.get("v9_shared_cls_fusion_weight", 1.0))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                stacked_logits = self._stack_calibrated_expert_logits(inputs, active_expert_ids)
                logits = self._fuse_calibrated_expert_logits(stacked_logits)

                if use_shared_cls:
                    shared_cls_logits = self._shared_cls_logits(inputs)
                    if shared_cls_logits is not None:
                        logits = expert_weight * logits + shared_weight * shared_cls_logits

            topk = min(self.topk, logits.shape[1])
            predicts = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1]
            if topk < self.topk:
                pad = torch.full(
                    (predicts.shape[0], self.topk - topk),
                    -1,
                    device=predicts.device,
                    dtype=predicts.dtype,
                )
                predicts = torch.cat([predicts, pad], dim=1)

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            _, _, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4)

            vectors = []
            for _, _inputs, _targets in idx_loader:
                vectors.append(model(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"])
            vectors = torch.cat(vectors, dim=0)

            self.cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
            covariance = torch.cov(vectors.T) + (torch.eye(self.cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
            if self.args["ca_storage_efficient_method"] == "covariance":
                self.cls_cov[class_idx] = covariance
            elif self.args["ca_storage_efficient_method"] == "variance":
                self.cls_cov[class_idx] = torch.diag(covariance)

    @torch.no_grad()
    def _compute_shared_cls_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            _, _, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4)

            vectors = []
            for _, _inputs, _targets in idx_loader:
                res = model(_inputs.to(self._device), adapter_id=-1, train=False)
                vectors.append(res["cls_features"])
            vectors = torch.cat(vectors, dim=0)

            self.shared_cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
            covariance = torch.cov(vectors.T) + (
                torch.eye(self.shared_cls_mean[class_idx].shape[-1]) * 1e-4
            ).to(self._device)
            if self.args["ca_storage_efficient_method"] == "covariance":
                self.shared_cls_cov[class_idx] = covariance
            elif self.args["ca_storage_efficient_method"] == "variance":
                self.shared_cls_cov[class_idx] = torch.diag(covariance)

    def classifer_align(self, model):
        del model
        for p in self._network.fc.parameters():
            p.requires_grad = True

        network_params = [
            {"params": self._network.fc.parameters(), "lr": self.ca_lr, "weight_decay": self.weight_decay}
        ]
        optimizer = optim.SGD(network_params, lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)

        prog_bar = tqdm(range(self.crct_epochs))
        task_size = self._known_classes - self._total_classes
        self._network.eval()
        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ["covariance", "variance"]:
                for class_idx in range(self._total_classes):
                    if self.args["decay"]:
                        t_id = class_idx // task_size
                        decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                        mean = torch.tensor(self.cls_mean[class_idx], dtype=torch.float64).to(self._device) * (
                            0.9 + decay
                        )
                    else:
                        mean = self.cls_mean[class_idx].to(self._device)
                    cov = self.cls_cov[class_idx].to(self._device)
                    distribution = self._build_safe_distribution(mean, cov)
                    sampled_data.append(distribution.sample(sample_shape=(num_sampled_pcls,)))
                    sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            shuffle_indices = torch.randperm(inputs.size(0))
            inputs = inputs[shuffle_indices]
            targets = targets[shuffle_indices]

            losses = 0.0
            correct, total = 0, 0
            for class_idx in range(self._total_classes):
                inp = inputs[class_idx * num_sampled_pcls : (class_idx + 1) * num_sampled_pcls]
                tgt = targets[class_idx * num_sampled_pcls : (class_idx + 1) * num_sampled_pcls]
                outputs = self._network.fc(inp)["logits"][:, : self._total_classes]
                logits = self.args["scale"] * outputs

                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = 100 * correct / total
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                losses / self._total_classes,
                train_acc,
            )
            prog_bar.set_description(info)

    def _classifier_align_module(self, classifier, mean_dict, cov_dict, stage, run_epochs, lr):
        if classifier is None:
            return

        for p in classifier.parameters():
            p.requires_grad = True

        network_params = [{"params": classifier.parameters(), "lr": lr, "weight_decay": self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)

        prog_bar = tqdm(range(run_epochs))
        task_size = self._known_classes - self._total_classes
        self._network.eval()
        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ["covariance", "variance"]:
                for class_idx in range(self._total_classes):
                    if self.args["decay"]:
                        t_id = class_idx // task_size
                        decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                        mean = torch.tensor(mean_dict[class_idx], dtype=torch.float64).to(self._device) * (0.9 + decay)
                    else:
                        mean = mean_dict[class_idx].to(self._device)
                    cov = cov_dict[class_idx].to(self._device)
                    distribution = self._build_safe_distribution(mean, cov)
                    sampled_data.append(distribution.sample(sample_shape=(num_sampled_pcls,)))
                    sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            shuffle_indices = torch.randperm(inputs.size(0))
            inputs = inputs[shuffle_indices]
            targets = targets[shuffle_indices]

            losses = 0.0
            correct, total = 0, 0
            for class_idx in range(self._total_classes):
                inp = inputs[class_idx * num_sampled_pcls : (class_idx + 1) * num_sampled_pcls]
                tgt = targets[class_idx * num_sampled_pcls : (class_idx + 1) * num_sampled_pcls]
                outputs = classifier(inp)["logits"]
                logits = self.args["scale"] * outputs

                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            train_acc = 100 * correct / total
            info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                stage,
                self._cur_task,
                epoch + 1,
                run_epochs,
                losses / self._total_classes,
                train_acc,
            )
            prog_bar.set_description(info)

    def _classifier_align_shared_cls(self):
        self._classifier_align_module(
            classifier=self._network.fc_shared_cls,
            mean_dict=self.shared_cls_mean,
            cov_dict=self.shared_cls_cov,
            stage="shared_cls_classifier_align",
            run_epochs=self.shared_cls_crct_epochs,
            lr=self.shared_cls_ca_lr,
        )
