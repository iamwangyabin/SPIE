import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import TunaLinear
from models.base import BaseLearner
from models.tuna import AngularPenaltySMLoss
from utils.inc_net import get_backbone
from utils.lion import Lion
from utils.toolkit import tensor2numpy

num_workers = 8


class TaskLocalCosineHead(nn.Module):
    def __init__(self, in_dim, out_dim, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x):
        cosine_logits = F.linear(
            F.normalize(x, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        logits = self.scale.to(device=x.device, dtype=x.dtype) * cosine_logits
        return {"cosine_logits": cosine_logits, "logits": logits}


class TaskLocalLinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return {"logits": self.fc(x)}


class SPIEV16Net(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = getattr(self.backbone, "out_dim", 768)
        self.fc_shared_cls = None
        self.expert_heads = nn.ModuleList()
        self.register_buffer("expert_energy_mean_in", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("expert_energy_std_in", torch.ones(0, dtype=torch.float32))
        self.local_head_scale = float(args.get("expert_local_head_scale", 10.0))
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def generate_fc(self, in_dim, out_dim):
        return TunaLinear(in_dim, out_dim)

    def update_fc(self, nb_classes):
        if self.fc_shared_cls is None:
            self.fc_shared_cls = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc_shared_cls.update(nb_classes, freeze_old=False)

    def append_expert_head(self, nb_classes):
        for head in self.expert_heads:
            head.requires_grad_(False)
        head = TaskLocalLinearHead(self.feature_dim, nb_classes).to(self._device)
        self.expert_heads.append(head)
        self.expert_energy_mean_in = torch.cat((self.expert_energy_mean_in, self.expert_energy_mean_in.new_zeros(1)))
        self.expert_energy_std_in = torch.cat((self.expert_energy_std_in, self.expert_energy_std_in.new_ones(1)))
        return head

    def get_expert_head(self, task_id):
        if task_id >= len(self.expert_heads):
            raise IndexError(f"Expert head {task_id} is not initialized.")
        return self.expert_heads[task_id]

    def freeze_expert_head(self, task_id):
        self.get_expert_head(task_id).requires_grad_(False)

    def get_expert_energy_stats(self, task_id):
        if task_id >= self.expert_energy_mean_in.shape[0]:
            raise IndexError(f"Expert energy stats for task {task_id} are not initialized.")
        return self.expert_energy_mean_in[task_id], self.expert_energy_std_in[task_id]

    @torch.no_grad()
    def set_expert_energy_stats(self, task_id, mean, std):
        if task_id >= self.expert_energy_mean_in.shape[0]:
            raise IndexError(f"Expert energy stats for task {task_id} are not initialized.")
        self.expert_energy_mean_in[task_id].copy_(
            torch.as_tensor(mean, dtype=self.expert_energy_mean_in.dtype, device=self.expert_energy_mean_in.device)
        )
        self.expert_energy_std_in[task_id].copy_(
            torch.as_tensor(std, dtype=self.expert_energy_std_in.dtype, device=self.expert_energy_std_in.device)
        )

    def forward_shared_cls(self, x, train=False):
        if self.fc_shared_cls is None:
            raise RuntimeError("fc_shared_cls is not initialized.")
        res = self.backbone(x, adapter_id=-1, train=train)
        cls_features = res["cls_features"]
        return {
            "cls_features": cls_features,
            "logits": self.fc_shared_cls(cls_features)["logits"],
        }

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        return self.backbone(x, adapter_id, train, fc_only)


class Learner(BaseLearner):
    """SPiE v16 with shared classification and energy-based expert task scoring."""

    _spie_version_name = "SPiE v16"

    def __init__(self, args):
        super().__init__(args)

        self._network = SPIEV16Net(args, True)
        self.shared_cls_mean = dict()
        self.shared_cls_cov = dict()
        self.task_class_ranges = []

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.args["tuned_epoch"] = args["tuned_epoch"]
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]

        self.share_lora_weight_decay = float(args.get("share_lora_weight_decay", self.weight_decay))
        self.expert_head_weight_decay = float(args.get("expert_head_weight_decay", self.weight_decay))

        self.task0_shared_epochs = int(args.get("task0_shared_epochs", args["tuned_epoch"]))
        self.task0_shared_lr = float(args.get("task0_shared_lr", self.init_lr * args.get("task0_shared_lr_scale", 1.0)))
        self.shared_cls_epochs = int(args.get("shared_cls_epochs", args["tuned_epoch"]))
        self.shared_cls_lr = float(args.get("shared_cls_lr", self.init_lr))
        self.shared_cls_weight_decay = float(args.get("shared_cls_weight_decay", self.weight_decay))
        self.shared_cls_ca_lr = float(args.get("shared_cls_ca_lr", self.ca_lr))
        self.shared_cls_crct_epochs = int(args.get("shared_cls_crct_epochs", self.crct_epochs))
        self.freeze_shared_lora_after_task0 = bool(args.get("freeze_shared_lora_after_task0", True))

        self.task0_expert_epochs = int(args.get("task0_expert_epochs", args["tuned_epoch"]))
        self.task0_expert_lr = float(args.get("task0_expert_lr", self.init_lr))
        self.incremental_expert_epochs = int(args.get("incremental_expert_epochs", args["tuned_epoch"]))
        self.incremental_expert_lr = float(
            args.get("incremental_expert_lr", self.init_lr * args.get("incremental_expert_lr_scale", 1.0))
        )
        self.expert_loss_type = str(args.get("expert_loss_type", "cosface")).lower()
        self.expert_loss_scale = float(args.get("expert_loss_scale", args.get("scale", 20.0)))
        self.expert_loss_margin = float(args.get("expert_loss_margin", args.get("m", 0.0)))
        self.energy_center_weight = float(args.get("energy_center_weight", 0.01))
        self.energy_scale_weight = float(args.get("energy_scale_weight", 0.01))
        self.energy_topk = max(int(args.get("energy_topk", 5)), 1)

        self.verifier_topk = min(int(args.get("verifier_topk", self.topk)), self.topk)
        self.verifier_local_topk = max(int(args.get("verifier_local_topk", 3)), 1)
        self.verifier_alpha = float(args.get("verifier_alpha", 0.5))
        self.verifier_align_epochs = int(args.get("verifier_align_epochs", 1))
        self.verifier_align_lr = float(args.get("verifier_align_lr", self.shared_cls_lr))
        self.verifier_align_weight = float(args.get("verifier_align_weight", 0.25))
        self.verifier_eps = float(args.get("verifier_eps", 1e-8))
        self.optimizer_grad_clip = float(args.get("optimizer_grad_clip", 0.0))

        for name, param in self._network.backbone.named_parameters():
            param.requires_grad = (
                "cur_adapter" in name
                or "cur_expert_tokens" in name
                or "cur_shared_adapter" in name
            )

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s %s total backbone parameters.", f"{total_params:,}", self._spie_version_name)
        logging.info("%s %s trainable backbone parameters.", f"{total_trainable_params:,}", self._spie_version_name)
        logging.info(
            "SPiE v16 shared branch: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, freeze_shared_lora_after_task0=%s.",
            self.task0_shared_epochs,
            self.task0_shared_lr,
            self.shared_cls_epochs,
            self.shared_cls_lr,
            self.freeze_shared_lora_after_task0,
        )
        logging.info(
            (
                "SPiE v16 expert energy: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, "
                "center_weight=%s, scale_weight=%s, energy_topk=%s, align_epochs=%s, align_weight=%s."
            ),
            self.task0_expert_epochs,
            self.task0_expert_lr,
            self.incremental_expert_epochs,
            self.incremental_expert_lr,
            self.energy_center_weight,
            self.energy_scale_weight,
            self.energy_topk,
            self.verifier_align_epochs,
            self.verifier_align_weight,
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
        current_task_size = self._total_classes - self._known_classes
        self.task_class_ranges.append((self._known_classes, self._total_classes))

        self._network.update_fc(current_task_size)
        self._network.append_expert_head(current_task_size)
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

        use_backbone_dataparallel = bool(self.args.get("spie_v15_backbone_dataparallel", False))
        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = nn.DataParallel(self._network.backbone, self._multiple_gpus)

        self._train(self.train_loader)

        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = self._backbone_module()

    def _make_optimizer(self, network_params):
        optimizer_name = str(self.args["optimizer"]).lower()
        if optimizer_name == "sgd":
            return optim.SGD(network_params, momentum=0.9)
        if optimizer_name == "adam":
            return optim.Adam(network_params)
        if optimizer_name == "adamw":
            return optim.AdamW(network_params)
        if optimizer_name in {"lion", "evolved_sign_momentum", "esm"}:
            lion_betas = tuple(self.args.get("lion_betas", (0.9, 0.99)))
            return Lion(network_params, betas=lion_betas)
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

    def _set_expert_head_requires_grad(self, task_id, requires_grad):
        self._network.get_expert_head(task_id).requires_grad_(requires_grad)

    def _optimizer_step(self, optimizer):
        if self.optimizer_grad_clip > 0:
            params = []
            for group in optimizer.param_groups:
                params.extend(group["params"])
            nn.utils.clip_grad_norm_(params, self.optimizer_grad_clip)
        optimizer.step()

    def _shared_branch_optimizer(self, lr):
        backbone = self._backbone_module()
        network_params = []
        shared_lora_params = [p for p in backbone.cur_shared_adapter.parameters() if p.requires_grad]
        if shared_lora_params:
            network_params.append(
                {
                    "params": shared_lora_params,
                    "lr": lr,
                    "weight_decay": self.share_lora_weight_decay,
                }
            )
        network_params.append(
            {
                "params": self._network.fc_shared_cls.parameters(),
                "lr": self.shared_cls_lr,
                "weight_decay": self.shared_cls_weight_decay,
            }
        )
        return self._make_optimizer(network_params)

    def _current_expert_optimizer(self, lr):
        backbone = self._backbone_module()
        expert_params = [
            p
            for name, p in backbone.named_parameters()
            if p.requires_grad and ("cur_adapter" in name or "cur_expert_tokens" in name)
        ]
        expert_head_params = [p for p in self._network.get_expert_head(self._cur_task).parameters() if p.requires_grad]
        network_params = [
            {
                "params": expert_params,
                "lr": lr,
                "weight_decay": self.weight_decay,
            },
            {
                "params": expert_head_params,
                "lr": lr,
                "weight_decay": self.expert_head_weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    def _class_to_task_id(self, class_idx):
        for task_id, (start, end) in enumerate(self.task_class_ranges):
            if start <= class_idx < end:
                return task_id
        raise ValueError(f"Class {class_idx} is not covered by any task range.")

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

    def _energy_from_logits(self, logits):
        return torch.logsumexp(logits, dim=-1)

    def _init_running_energy_stats(self):
        return {
            "count": 0,
            "mean": torch.zeros((), device=self._device, dtype=torch.float32),
            "m2": torch.zeros((), device=self._device, dtype=torch.float32),
        }

    def _update_running_energy_stats(self, stats, values):
        values = values.detach().reshape(-1).to(device=self._device, dtype=torch.float32)
        if values.numel() == 0:
            return stats

        batch_count = values.numel()
        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_m2 = batch_var * batch_count

        if stats["count"] == 0:
            stats["count"] = batch_count
            stats["mean"] = batch_mean
            stats["m2"] = batch_m2
            return stats

        total_count = stats["count"] + batch_count
        delta = batch_mean - stats["mean"]
        stats["mean"] = stats["mean"] + delta * batch_count / total_count
        stats["m2"] = stats["m2"] + batch_m2 + delta.pow(2) * stats["count"] * batch_count / total_count
        stats["count"] = total_count
        return stats

    def _finalize_running_energy_stats(self, stats):
        if stats["count"] <= 0:
            return 0.0, 1.0

        variance = stats["m2"] / stats["count"]
        std = torch.sqrt(variance.clamp_min(1e-12))
        return stats["mean"].detach(), std.detach()

    def _zscore_tensor(self, values, dim=-1, eps=1e-6):
        if values.shape[dim] <= 1:
            return torch.zeros_like(values)
        mean = values.mean(dim=dim, keepdim=True)
        std = values.std(dim=dim, keepdim=True, unbiased=False)
        return (values - mean) / std.clamp_min(eps)

    def _collect_expert_logits(self, inputs, task_ids):
        if not task_ids:
            return {}

        task_ids = list(task_ids)
        backbone = self._backbone_module()
        if len(task_ids) > 1 and hasattr(backbone, "forward_multi_expert_features"):
            res = backbone.forward_multi_expert_features(inputs, task_ids)
            expert_feature_map = {
                task_id: res["expert_features"][local_idx]
                for local_idx, task_id in enumerate(task_ids)
            }
        else:
            expert_feature_map = {
                task_id: self._network.backbone(inputs, adapter_id=task_id, train=False)["expert_features"]
                for task_id in task_ids
            }

        return {
            task_id: self._network.get_expert_head(task_id)(expert_feature_map[task_id])["logits"]
            for task_id in task_ids
        }

    def _train(self, train_loader):
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc_shared_cls.to(self._device)
        self._network.expert_heads.to(self._device)

        if self._cur_task == 0:
            self._train_shared_branch(
                train_loader=train_loader,
                epochs=self.task0_shared_epochs,
                branch_lr=self.task0_shared_lr,
                stage="task0_shared_branch",
            )
            self._train_current_expert(
                train_loader=train_loader,
                epochs=self.task0_expert_epochs,
                expert_lr=self.task0_expert_lr,
                stage="task0_expert_local",
            )
        else:
            self._train_shared_branch(
                train_loader=train_loader,
                epochs=self.shared_cls_epochs,
                branch_lr=self.shared_cls_lr,
                stage="shared_branch",
            )
            self._train_current_expert(
                train_loader=train_loader,
                epochs=self.incremental_expert_epochs,
                expert_lr=self.incremental_expert_lr,
                stage="incremental_expert_local",
            )

        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(False)
        self._set_expert_head_requires_grad(self._cur_task, False)
        backbone_module.adapter_update()

        self._train_shared_expert_alignment(
            train_loader=train_loader,
            epochs=self.verifier_align_epochs,
            align_lr=self.verifier_align_lr,
            stage="verifier_alignment",
        )

        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

    def _train_shared_branch(self, train_loader, epochs, branch_lr, stage):
        if epochs <= 0 or self._network.fc_shared_cls is None:
            return

        train_shared_lora = not self.freeze_shared_lora_after_task0 or self._cur_task == 0
        self._set_shared_lora_requires_grad(train_shared_lora)
        self._set_current_expert_requires_grad(False)
        self._set_expert_head_requires_grad(self._cur_task, False)

        optimizer = self._shared_branch_optimizer(branch_lr)
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.fc_shared_cls.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                cls_features = self._network.backbone(inputs, adapter_id=-1, train=True)["cls_features"]
                logits = self._network.fc_shared_cls(cls_features)["logits"]

                loss = loss_cos(logits[:, self._known_classes : self._total_classes], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                self._optimizer_step(optimizer)

                losses += loss.item()
                _, preds = torch.max(logits[:, self._known_classes : self._total_classes], dim=1)
                preds = preds + self._known_classes
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
                shared_lora_trainable=bool(train_shared_lora),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _train_current_expert(self, train_loader, epochs, expert_lr, stage):
        if epochs <= 0:
            return

        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(True)
        self._set_expert_head_requires_grad(self._cur_task, True)

        optimizer = self._current_expert_optimizer(expert_lr)
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        expert_head = self._network.get_expert_head(self._cur_task)
        running_energy_stats = self._init_running_energy_stats()
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            expert_head.train()
            losses = 0.0
            ce_losses = 0.0
            center_losses = 0.0
            scale_losses = 0.0
            batch_energy_means = 0.0
            batch_energy_stds = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes
                expert_features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["expert_features"]
                expert_out = expert_head(expert_features)
                logits = expert_out["logits"]
                energy_scores = self._energy_from_logits(logits)
                ce_loss = F.cross_entropy(logits, local_targets)
                energy_center_loss = energy_scores.mean().pow(2)
                energy_scale_loss = (energy_scores.std(unbiased=False) - 1.0).pow(2)
                loss = (
                    ce_loss
                    + self.energy_center_weight * energy_center_loss
                    + self.energy_scale_weight * energy_scale_loss
                )

                optimizer.zero_grad()
                loss.backward()
                self._optimizer_step(optimizer)
                running_energy_stats = self._update_running_energy_stats(running_energy_stats, energy_scores)

                losses += loss.item()
                ce_losses += ce_loss.item()
                center_losses += energy_center_loss.item()
                scale_losses += energy_scale_loss.item()
                batch_energy_means += energy_scores.mean().item()
                batch_energy_stds += energy_scores.std(unbiased=False).item()
                preds = torch.argmax(logits, dim=1) + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            avg_ce_loss = ce_losses / len(train_loader)
            avg_center_loss = center_losses / len(train_loader)
            avg_scale_loss = scale_losses / len(train_loader)
            avg_energy_mean = batch_energy_means / len(train_loader)
            avg_energy_std = batch_energy_stds / len(train_loader)
            info = "Task {}, {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                stage,
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                stage=stage,
                ce_loss=float(avg_ce_loss),
                energy_center_loss=float(avg_center_loss),
                energy_scale_loss=float(avg_scale_loss),
                batch_energy_mean=float(avg_energy_mean),
                batch_energy_std=float(avg_energy_std),
            )
            prog_bar.set_description(info)

        energy_mean_in, energy_std_in = self._finalize_running_energy_stats(running_energy_stats)
        self._network.set_expert_energy_stats(self._cur_task, energy_mean_in, energy_std_in)
        logging.info(
            "Task %s expert energy stats saved: mean=%.4f std=%.4f",
            self._cur_task,
            float(torch.as_tensor(energy_mean_in).item()),
            float(torch.as_tensor(energy_std_in).item()),
        )
        logging.info(info)

    def _train_shared_expert_alignment(self, train_loader, epochs, align_lr, stage):
        if epochs <= 0 or self.verifier_align_weight <= 0:
            return

        backbone_module = self._backbone_module()
        train_shared_lora = not self.freeze_shared_lora_after_task0 or self._cur_task == 0
        self._set_shared_lora_requires_grad(train_shared_lora)
        self._set_current_expert_requires_grad(False)
        self._set_expert_head_requires_grad(self._cur_task, False)

        optimizer = self._shared_branch_optimizer(align_lr)
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])
        expert_head = self._network.get_expert_head(self._cur_task)

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.fc_shared_cls.train()
            expert_head.eval()
            if self._cur_task < len(backbone_module.adapter_list):
                backbone_module.adapter_list[self._cur_task].eval()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes

                cls_features = self._network.backbone(inputs, adapter_id=-1, train=True)["cls_features"]
                shared_logits = self._network.fc_shared_cls(cls_features)["logits"]
                shared_task_logits = shared_logits[:, self._known_classes : self._total_classes]

                with torch.no_grad():
                    expert_features = self._network.backbone(inputs, adapter_id=self._cur_task, train=False)["expert_features"]
                    expert_logits = expert_head(expert_features)["logits"]

                ce_loss = loss_cos(shared_task_logits, local_targets)
                align_loss = self._batch_task_local_js(shared_task_logits, expert_logits).mean()
                loss = ce_loss + self.verifier_align_weight * align_loss

                optimizer.zero_grad()
                loss.backward()
                self._optimizer_step(optimizer)

                losses += loss.item()
                preds = torch.argmax(shared_task_logits, dim=1) + self._known_classes
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
                align_weight=float(self.verifier_align_weight),
                shared_lora_trainable=bool(train_shared_lora),
            )
            prog_bar.set_description(info)

        logging.info(info)
        self._set_shared_lora_requires_grad(False)

    def _support_union_indices(self, shared_slice, expert_slice):
        shared_k = min(self.verifier_local_topk, shared_slice.numel())
        expert_k = min(self.verifier_local_topk, expert_slice.numel())
        shared_idx = torch.topk(shared_slice, k=shared_k, dim=0, largest=True, sorted=False).indices
        expert_idx = torch.topk(expert_slice, k=expert_k, dim=0, largest=True, sorted=False).indices
        return torch.unique(torch.cat((shared_idx, expert_idx), dim=0), sorted=True)

    def _task_local_js(self, shared_slice, expert_slice):
        if shared_slice.numel() == 0 or expert_slice.numel() == 0:
            return shared_slice.new_zeros(())

        support = self._support_union_indices(shared_slice, expert_slice)
        shared_local = F.softmax(shared_slice[support], dim=0)
        expert_local = F.softmax(expert_slice[support], dim=0)
        midpoint = 0.5 * (shared_local + expert_local)

        shared_log = torch.log(shared_local.clamp_min(self.verifier_eps))
        expert_log = torch.log(expert_local.clamp_min(self.verifier_eps))
        midpoint_log = torch.log(midpoint.clamp_min(self.verifier_eps))
        shared_kl = torch.sum(shared_local * (shared_log - midpoint_log))
        expert_kl = torch.sum(expert_local * (expert_log - midpoint_log))
        return 0.5 * (shared_kl + expert_kl)

    def _batch_task_local_js(self, shared_slices, expert_slices):
        js_values = []
        for sample_idx in range(shared_slices.shape[0]):
            js_values.append(self._task_local_js(shared_slices[sample_idx], expert_slices[sample_idx]))
        return torch.stack(js_values, dim=0)

    def _batch_task_local_similarity(self, shared_slices, expert_slices):
        return torch.exp(-self._batch_task_local_js(shared_slices, expert_slices))

    def _shared_cls_logits(self, inputs):
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        return self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]

    def _predict_topk(self, logits):
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
        return predicts

    def _select_candidate_tasks(self, topk_indices):
        candidate_tasks = []
        unique_task_ids = []
        seen_global = set()

        for row in topk_indices.tolist():
            row_tasks = []
            row_seen = set()
            for class_idx in row[: self.energy_topk]:
                if class_idx < 0:
                    continue
                task_id = self._class_to_task_id(int(class_idx))
                if task_id not in row_seen:
                    row_seen.add(task_id)
                    row_tasks.append(task_id)
                if task_id not in seen_global:
                    seen_global.add(task_id)
                    unique_task_ids.append(task_id)
            candidate_tasks.append(row_tasks)

        return candidate_tasks, unique_task_ids

    def _compute_task_local_similarities(self, inputs, shared_logits, topk_indices):
        _, unique_task_ids = self._select_candidate_tasks(topk_indices)
        if not unique_task_ids:
            return {}

        task_similarities = {}
        backbone = self._backbone_module()
        if len(unique_task_ids) > 1 and hasattr(backbone, "forward_multi_expert_features"):
            res = backbone.forward_multi_expert_features(inputs, unique_task_ids)
            expert_feature_map = {
                task_id: res["expert_features"][local_idx]
                for local_idx, task_id in enumerate(unique_task_ids)
            }
        else:
            expert_feature_map = {}
            for task_id in unique_task_ids:
                expert_feature_map[task_id] = self._network.backbone(inputs, adapter_id=task_id, train=False)[
                    "expert_features"
                ]

        for task_id in unique_task_ids:
            start_idx, end_idx = self.task_class_ranges[task_id]
            shared_slice = shared_logits[:, start_idx:end_idx]
            expert_logits = self._network.get_expert_head(task_id)(expert_feature_map[task_id])["logits"]
            task_similarities[task_id] = self._batch_task_local_similarity(shared_slice, expert_logits)

        return task_similarities

    def _rerank_topk(self, shared_logits, topk_indices, task_similarities):
        reranked = topk_indices.clone()
        candidate_width = min(self.verifier_topk, topk_indices.shape[1])

        for sample_idx in range(topk_indices.shape[0]):
            candidate_classes = topk_indices[sample_idx, :candidate_width]
            valid_mask = candidate_classes >= 0
            valid_classes = candidate_classes[valid_mask]
            if valid_classes.numel() == 0:
                continue

            adjusted_scores = []
            for class_idx in valid_classes.tolist():
                class_idx = int(class_idx)
                task_id = self._class_to_task_id(class_idx)
                bonus = 0.0
                if task_id in task_similarities:
                    bonus = self.verifier_alpha * task_similarities[task_id][sample_idx]
                adjusted_scores.append(shared_logits[sample_idx, class_idx] + bonus)

            adjusted_scores = torch.stack(adjusted_scores, dim=0)
            order = torch.argsort(adjusted_scores, descending=True)
            reranked[sample_idx, :candidate_width] = candidate_classes
            reranked[sample_idx, : valid_classes.numel()] = valid_classes[order]

        return reranked

    def eval_task(self):
        cnn_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(cnn_pred, y_true)

        nme_pred, nme_true = self._eval_nme(self.test_loader, class_means=None)
        nme_accy = self._evaluate(nme_pred, nme_true)

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                logits = self._shared_cls_logits(inputs)
                predicts = self._predict_topk(logits)

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        del class_means

        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                gt_task_ids = [self._class_to_task_id(int(target.item())) for target in targets]
                unique_task_ids = sorted(set(gt_task_ids))
                expert_logits_map = self._collect_expert_logits(inputs, unique_task_ids)

                predicts = torch.full(
                    (targets.shape[0], self.topk),
                    -1,
                    device=inputs.device,
                    dtype=torch.long,
                )

                # NME for SPiE v16 is defined as oracle expert-local classification:
                # use the ground-truth task's expert and rank classes only within that expert.
                for sample_idx, task_id in enumerate(gt_task_ids):
                    start_idx, _ = self.task_class_ranges[task_id]
                    expert_slice = expert_logits_map[task_id][sample_idx]
                    final_order = torch.argsort(expert_slice, descending=True)
                    local_topk = min(self.topk, final_order.numel())
                    predicts[sample_idx, :local_topk] = final_order[:local_topk] + start_idx

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

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
            else:
                raise NotImplementedError

    def _classifier_align_module(self, classifier, mean_dict, cov_dict, stage, run_epochs, lr):
        if classifier is None or run_epochs <= 0:
            return

        for p in classifier.parameters():
            p.requires_grad = True

        network_params = [{"params": classifier.parameters(), "lr": lr, "weight_decay": self.weight_decay}]
        optimizer = self._make_optimizer(network_params)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max(run_epochs, 1))

        prog_bar = tqdm(range(run_epochs))
        self._network.eval()
        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ["covariance", "variance"]:
                for class_idx in range(self._total_classes):
                    if self.args["decay"]:
                        task_id = self._class_to_task_id(class_idx)
                        decay = (task_id + 1) / (self._cur_task + 1) * 0.1
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
                start_idx = class_idx * num_sampled_pcls
                end_idx = (class_idx + 1) * num_sampled_pcls
                inp = inputs[start_idx:end_idx]
                tgt = targets[start_idx:end_idx]
                outputs = classifier(inp)["logits"]
                logits = self.args["scale"] * outputs

                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                self._optimizer_step(optimizer)
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
