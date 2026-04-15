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
from utils.toolkit import tensor2numpy

num_workers = 8


class TaskOODDetector(nn.Module):
    """Prototype-based task scorer built on expert feature banks."""

    def __init__(self, feature_dim, hidden_dim=16, topk=5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.topk = max(int(topk), 1)
        self.hidden_dim = int(hidden_dim)
        self.stat_dim = 6
        self.ood_score_topk = max(int(topk), 1)

        self.register_buffer("class_centers", torch.zeros(0, self.feature_dim))
        self.register_buffer("class_diag_vars", torch.zeros(0, self.feature_dim))
        self.register_buffer("representatives", torch.zeros(0, self.feature_dim))
        self.register_buffer("positive_bank", torch.zeros(0, self.feature_dim))
        self.register_buffer("ood_bank", torch.zeros(0, self.feature_dim))
        self.score_eps = 1e-6

    @staticmethod
    def _normalize(x):
        return F.normalize(x, p=2, dim=-1)

    @torch.no_grad()
    def set_feature_bank(self, class_centers, class_diag_vars, representatives):
        if class_centers is None:
            class_centers = torch.zeros(0, self.feature_dim, device=self.class_centers.device)
        if class_diag_vars is None:
            class_diag_vars = torch.zeros(0, self.feature_dim, device=self.class_centers.device)
        if representatives is None:
            representatives = torch.zeros(0, self.feature_dim, device=self.class_centers.device)

        class_centers = class_centers.detach().to(device=self.class_centers.device, dtype=self.class_centers.dtype)
        class_diag_vars = class_diag_vars.detach().to(device=self.class_centers.device, dtype=self.class_centers.dtype)
        representatives = representatives.detach().to(device=self.class_centers.device, dtype=self.class_centers.dtype)

        if class_centers.ndim == 1:
            class_centers = class_centers.unsqueeze(0)
        if class_diag_vars.ndim == 1:
            class_diag_vars = class_diag_vars.unsqueeze(0)
        if representatives.ndim == 1:
            representatives = representatives.unsqueeze(0)

        if class_centers.numel() > 0:
            class_centers = self._normalize(class_centers)
        if class_diag_vars.numel() > 0:
            class_diag_vars = torch.clamp(class_diag_vars, min=self.score_eps)
        if representatives.numel() > 0:
            representatives = self._normalize(representatives)

        positive_bank = class_centers
        if representatives.numel() > 0:
            positive_bank = (
                torch.cat((positive_bank, representatives), dim=0)
                if positive_bank.numel() > 0
                else representatives
            )

        self.class_centers = class_centers
        self.class_diag_vars = class_diag_vars
        self.representatives = representatives
        self.positive_bank = positive_bank

    @torch.no_grad()
    def set_ood_bank(self, ood_bank):
        if ood_bank is None:
            ood_bank = torch.zeros(0, self.feature_dim, device=self.class_centers.device)
        ood_bank = ood_bank.detach().to(device=self.class_centers.device, dtype=self.class_centers.dtype)
        if ood_bank.ndim == 1:
            ood_bank = ood_bank.unsqueeze(0)
        if ood_bank.numel() > 0:
            ood_bank = self._normalize(ood_bank)
        self.ood_bank = ood_bank

    def compute_stats(self, features):
        features = features.float()
        normed_features = self._normalize(features)
        batch_size = normed_features.shape[0]
        device = normed_features.device

        if self.class_centers.numel() > 0:
            centers = self.class_centers.to(device=device, dtype=normed_features.dtype)
            center_sims = normed_features @ centers.T
            max_center = center_sims.max(dim=1).values
            mean_center = center_sims.mean(dim=1)
        else:
            max_center = torch.zeros(batch_size, device=device, dtype=normed_features.dtype)
            mean_center = torch.zeros_like(max_center)

        if self.representatives.numel() > 0:
            representatives = self.representatives.to(device=device, dtype=normed_features.dtype)
            rep_sims = normed_features @ representatives.T
            topk = min(self.topk, rep_sims.shape[1])
            topk_sims = torch.topk(rep_sims, k=topk, dim=1, largest=True, sorted=True).values
            rep_max = topk_sims[:, 0]
            rep_mean = topk_sims.mean(dim=1)
            rep_global_mean = rep_sims.mean(dim=1)
        else:
            rep_max = torch.zeros(batch_size, device=device, dtype=normed_features.dtype)
            rep_mean = torch.zeros_like(rep_max)
            rep_global_mean = torch.zeros_like(rep_max)

        if self.ood_bank.numel() > 0:
            ood_bank = self.ood_bank.to(device=device, dtype=normed_features.dtype)
            ood_sims = normed_features @ ood_bank.T
            ood_topk = min(self.ood_score_topk, ood_sims.shape[1])
            ood_penalty = torch.topk(ood_sims, k=ood_topk, dim=1, largest=True, sorted=True).values.mean(dim=1)
        else:
            ood_penalty = torch.zeros(batch_size, device=device, dtype=normed_features.dtype)

        if self.class_centers.numel() > 0 and self.class_diag_vars.numel() > 0:
            centers = self.class_centers.to(device=device, dtype=normed_features.dtype)
            diag_vars = self.class_diag_vars.to(device=device, dtype=normed_features.dtype)
            diff = normed_features.unsqueeze(1) - centers.unsqueeze(0)
            maha = (diff.pow(2) / torch.clamp(diag_vars.unsqueeze(0), min=self.score_eps)).mean(dim=2)
            min_maha = maha.min(dim=1).values
            topk_maha = torch.topk(
                maha,
                k=min(self.topk, maha.shape[1]),
                dim=1,
                largest=False,
                sorted=True,
            ).values
            mean_topk_maha = topk_maha.mean(dim=1)
        else:
            min_maha = torch.full((batch_size,), 1e3, device=device, dtype=normed_features.dtype)
            mean_topk_maha = min_maha

        return torch.stack(
            [max_center, mean_center, rep_max, rep_mean, -min_maha, ood_penalty],
            dim=1,
        )

    def score_from_stats(self, stats):
        return stats[:, 2] + stats[:, 4] - stats[:, 5]

    def forward(self, features):
        stats = self.compute_stats(features)
        return {
            "stats": stats,
            "scores": self.score_from_stats(stats),
        }


class SPIEV14Net(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = getattr(self.backbone, "out_dim", 768)
        self.fc_shared_cls = None
        self.task_ood_detectors = nn.ModuleList()
        self.task_ood_detector_hidden_dim = int(args.get("task_ood_detector_hidden_dim", 16))
        self.task_ood_detector_topk = int(args.get("task_ood_detector_topk", 5))
        self.hard_ood_score_topk = int(args.get("hard_ood_score_topk", self.task_ood_detector_topk))
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
        self._ensure_task_ood_detectors()

    def _ensure_task_ood_detectors(self):
        if self.fc_shared_cls is None or not hasattr(self.fc_shared_cls, "heads"):
            return

        target_heads = len(self.fc_shared_cls.heads)
        while len(self.task_ood_detectors) < target_heads:
            self.task_ood_detectors.append(
                TaskOODDetector(
                    feature_dim=self.feature_dim,
                    hidden_dim=self.task_ood_detector_hidden_dim,
                    topk=self.task_ood_detector_topk,
                ).to(self._device)
            )
            self.task_ood_detectors[-1].ood_score_topk = max(self.hard_ood_score_topk, 1)

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

    def get_task_ood_detector(self, task_id):
        if task_id >= len(self.task_ood_detectors):
            raise IndexError(f"Task OOD detector {task_id} is not initialized.")
        return self.task_ood_detectors[task_id]

    @torch.no_grad()
    def set_task_ood_feature_bank(self, task_id, class_centers, class_diag_vars, representatives):
        detector = self.get_task_ood_detector(task_id)
        detector.set_feature_bank(class_centers, class_diag_vars, representatives)

    def forward_multi_expert_ood_scores(self, x, expert_ids):
        backbone = self.backbone.module if isinstance(self.backbone, nn.DataParallel) else self.backbone
        if not hasattr(backbone, "forward_multi_expert_features"):
            raise RuntimeError("Current backbone does not support parallel multi-expert inference.")

        res = backbone.forward_multi_expert_features(x, expert_ids)
        expert_features = res["expert_features"]
        scores = []
        stats = []
        for local_idx, task_id in enumerate(expert_ids):
            detector_out = self.get_task_ood_detector(task_id)(expert_features[local_idx])
            scores.append(detector_out["scores"])
            stats.append(detector_out["stats"])

        return {
            "cls_features": res["cls_features"],
            "expert_features": expert_features,
            "scores": torch.stack(scores, dim=0),
            "stats": torch.stack(stats, dim=0),
        }


class Learner(BaseLearner):
    """SPiE v14 with shared-CLS classification and expert task-prior detectors."""

    _spie_version_name = "SPiE v14"

    def __init__(self, args):
        super().__init__(args)

        self._network = SPIEV14Net(args, True)
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
        self.expert_prototype_temperature = float(args.get("expert_prototype_temperature", 0.1))
        self.expert_compactness_weight = float(args.get("expert_compactness_weight", 0.05))

        self.ood_repr_per_class = int(args.get("ood_repr_per_class", 8))
        self.in_task_repr_per_class = int(args.get("in_task_repr_per_class", max(self.ood_repr_per_class, 24)))
        self.hard_ood_per_task = int(args.get("hard_ood_per_task", 16))
        self.hard_ood_score_topk = int(args.get("hard_ood_score_topk", 4))
        self.ood_calibration_epochs = int(args.get("ood_calibration_epochs", 5))
        self.ood_calibration_lr = float(args.get("ood_calibration_lr", self.init_lr * 0.1))
        self.ood_calibration_weight_decay = float(args.get("ood_calibration_weight_decay", 0.0))
        self.ood_score_temperature = float(args.get("ood_score_temperature", 1.0))
        self.ood_weight_alpha = float(args.get("ood_weight_alpha", 0.2))
        self._expert_ood_task_memory = {}

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
            "SPiE v14 shared branch: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, freeze_shared_lora_after_task0=%s.",
            self.task0_shared_epochs,
            self.task0_shared_lr,
            self.shared_cls_epochs,
            self.shared_cls_lr,
            self.freeze_shared_lora_after_task0,
        )
        logging.info(
            "SPiE v14 expert OOD branch: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, calib epochs=%s lr=%s.",
            self.task0_expert_epochs,
            self.task0_expert_lr,
            self.incremental_expert_epochs,
            self.incremental_expert_lr,
            self.ood_calibration_epochs,
            self.ood_calibration_lr,
        )
        logging.info(
            "SPiE v14 memory: in_task_repr_per_class=%s, hard_ood_per_task=%s, hard_ood_score_topk=%s.",
            self.in_task_repr_per_class,
            self.hard_ood_per_task,
            self.hard_ood_score_topk,
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
        self.task_class_ranges.append((self._known_classes, self._total_classes))

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

        use_backbone_dataparallel = bool(self.args.get("spie_v14_backbone_dataparallel", False))
        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = nn.DataParallel(self._network.backbone, self._multiple_gpus)

        self._train(self.train_loader)

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
        network_params = [
            {
                "params": expert_params,
                "lr": lr,
                "weight_decay": self.weight_decay,
            }
        ]
        return self._make_optimizer(network_params)

    @staticmethod
    def _restore_requires_grad(params, flags):
        for param, requires_grad in zip(params, flags):
            param.requires_grad = requires_grad

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

    def _train(self, train_loader):
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc_shared_cls.to(self._device)
        self._network.task_ood_detectors.to(self._device)

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
                stage="task0_expert_ood",
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
                stage="incremental_expert_ood",
            )

        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(True)
        backbone_module.adapter_update()
        self._update_current_task_ood_bank(backbone)
        self._update_historical_ood_detectors(train_loader)

        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

    def _train_shared_branch(self, train_loader, epochs, branch_lr, stage):
        if epochs <= 0 or self._network.fc_shared_cls is None:
            return

        train_shared_lora = not self.freeze_shared_lora_after_task0 or self._cur_task == 0
        self._set_shared_lora_requires_grad(train_shared_lora)
        self._set_current_expert_requires_grad(False)

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

    def _expert_prototype_loss(self, features, targets):
        features = F.normalize(features, p=2, dim=1)
        unique_targets = torch.unique(targets, sorted=True)
        prototypes = []
        local_target_map = {}
        for local_idx, class_idx in enumerate(unique_targets.tolist()):
            class_mask = targets == class_idx
            prototype = features[class_mask].mean(dim=0)
            prototypes.append(F.normalize(prototype.unsqueeze(0), p=2, dim=1).squeeze(0))
            local_target_map[class_idx] = local_idx

        prototypes = torch.stack(prototypes, dim=0)
        local_targets = torch.tensor(
            [local_target_map[int(class_idx)] for class_idx in targets.tolist()],
            device=targets.device,
            dtype=torch.long,
        )
        logits = (features @ prototypes.T) / max(self.expert_prototype_temperature, 1e-6)
        ce_loss = F.cross_entropy(logits, local_targets)
        compactness = 1.0 - (features * prototypes[local_targets]).sum(dim=1).mean()
        loss = ce_loss + self.expert_compactness_weight * compactness
        return loss, logits, unique_targets

    def _train_current_expert(self, train_loader, epochs, expert_lr, stage):
        if epochs <= 0:
            return

        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(True)

        optimizer = self._current_expert_optimizer(expert_lr)
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                expert_features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["expert_features"]
                loss, logits, unique_targets = self._expert_prototype_loss(expert_features, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                local_preds = torch.argmax(logits, dim=1)
                preds = unique_targets[local_preds]
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
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                stage=stage,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _active_expert_ids(self):
        return list(range(self._cur_task + 1))

    @torch.no_grad()
    def _update_current_task_ood_bank(self, model):
        model.eval()
        class_centers = []
        class_diag_vars = []
        representatives = []

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
                res = model(_inputs.to(self._device), adapter_id=self._cur_task, train=False)
                vectors.append(F.normalize(res["expert_features"], p=2, dim=1))
            vectors = torch.cat(vectors, dim=0)

            center = F.normalize(vectors.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
            diag_var = torch.var(vectors, dim=0, unbiased=False) + 1e-4
            class_centers.append(center)
            class_diag_vars.append(diag_var)

            rep_count = min(self.in_task_repr_per_class, vectors.shape[0])
            if rep_count > 0:
                similarities = vectors @ center.unsqueeze(1)
                topk_indices = torch.topk(similarities.squeeze(1), k=rep_count, largest=True, sorted=True).indices
                representatives.append(vectors[topk_indices])

        class_centers = torch.stack(class_centers, dim=0) if class_centers else torch.zeros(0, self.feature_dim)
        class_diag_vars = torch.stack(class_diag_vars, dim=0) if class_diag_vars else torch.zeros(0, self.feature_dim)
        representatives = torch.cat(representatives, dim=0) if representatives else torch.zeros(0, self.feature_dim)
        self._network.set_task_ood_feature_bank(self._cur_task, class_centers, class_diag_vars, representatives)

    def _sample_positive_features(self, detector, num_samples):
        positive_bank = detector.positive_bank
        if positive_bank.numel() == 0:
            return None
        indices = torch.randint(0, positive_bank.shape[0], (num_samples,), device=positive_bank.device)
        return positive_bank[indices]

    @torch.no_grad()
    def _rebuild_expert_ood_bank(self, expert_id):
        task_memory = self._expert_ood_task_memory.get(expert_id, {})
        if task_memory:
            ordered_tasks = sorted(task_memory.keys())
            ood_bank = torch.cat([task_memory[task_id] for task_id in ordered_tasks], dim=0)
        else:
            ood_bank = torch.zeros(0, self.feature_dim, device=self._device)
        self._network.get_task_ood_detector(expert_id).set_ood_bank(ood_bank)

    @torch.no_grad()
    def _collect_hard_ood_for_expert(self, expert_id, negative_loader):
        detector = self._network.get_task_ood_detector(expert_id)
        candidates = []
        scores = []

        self._network.backbone.eval()
        for _, (_, inputs, _) in enumerate(negative_loader):
            inputs = inputs.to(self._device)
            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["expert_features"]
            features = F.normalize(features, p=2, dim=1)
            feature_scores = detector.score_from_stats(detector.compute_stats(features))
            candidates.append(features.detach())
            scores.append(feature_scores.detach())

        if not candidates:
            return torch.zeros(0, self.feature_dim, device=self._device)

        candidates = torch.cat(candidates, dim=0)
        scores = torch.cat(scores, dim=0)
        keep = min(self.hard_ood_per_task, candidates.shape[0])
        if keep <= 0:
            return torch.zeros(0, self.feature_dim, device=self._device)
        topk_indices = torch.topk(scores, k=keep, largest=True, sorted=True).indices
        return candidates[topk_indices]

    def _update_historical_ood_detectors(self, train_loader):
        if self._cur_task <= 0 or self.hard_ood_per_task <= 0:
            return

        for expert_id in range(self._cur_task):
            hard_ood = self._collect_hard_ood_for_expert(expert_id, train_loader)
            expert_memory = self._expert_ood_task_memory.setdefault(expert_id, {})
            expert_memory[self._cur_task] = hard_ood
            self._rebuild_expert_ood_bank(expert_id)

    def _train_single_ood_detector(self, expert_id, negative_loader):
        del expert_id, negative_loader
        return

    def _shared_cls_logits(self, inputs):
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        return self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]

    def _compute_task_prior_weights(self, task_scores):
        task_scores = task_scores / max(self.ood_score_temperature, 1e-6)
        return F.softmax(task_scores, dim=0)

    def _apply_task_prior(self, shared_logits, task_weights, task_ids):
        adjusted_logits = shared_logits.clone()
        uniform_prior = 1.0 / max(len(task_ids), 1)
        task_scales = 1.0 + self.ood_weight_alpha * (task_weights - uniform_prior)
        task_scales = torch.clamp(task_scales, min=0.5, max=1.5)

        for local_idx, task_id in enumerate(task_ids):
            start_idx, end_idx = self.task_class_ranges[task_id]
            adjusted_logits[:, start_idx:end_idx] = adjusted_logits[:, start_idx:end_idx] * task_scales[
                local_idx
            ].unsqueeze(1)

        return adjusted_logits

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

    def _expert_weighted_logits(self, inputs, active_expert_ids):
        shared_logits = self._shared_cls_logits(inputs)
        if not active_expert_ids:
            return shared_logits

        ood_out = self._network.forward_multi_expert_ood_scores(inputs, active_expert_ids)
        task_weights = self._compute_task_prior_weights(ood_out["scores"])
        return self._apply_task_prior(shared_logits, task_weights, active_expert_ids)

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
        active_expert_ids = self._active_expert_ids()

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                logits = self._expert_weighted_logits(inputs, active_expert_ids)
                predicts = self._predict_topk(logits)

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
        optimizer = optim.SGD(network_params, lr=lr, momentum=0.9, weight_decay=5e-4)
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
