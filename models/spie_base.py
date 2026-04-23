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


class SPIEBaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = getattr(self.backbone, "out_dim", 768)
        self.fc_shared_cls = None
        self.expert_heads = nn.ModuleList()
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    @property
    def expert_feature_dim(self):
        return self.backbone.out_dim * 2

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
        head = self.generate_fc(self.expert_feature_dim, nb_classes).to(self._device)
        self.expert_heads.append(head)
        return head

    def get_expert_head(self, task_id):
        if task_id >= len(self.expert_heads):
            raise IndexError(f"Expert head {task_id} is not initialized.")
        return self.expert_heads[task_id]

    def freeze_expert_head(self, task_id):
        self.get_expert_head(task_id).requires_grad_(False)

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
    """Shared SPiE backbone/head implementation."""

    _spie_version_name = "SPiE"

    def __init__(self, args):
        super().__init__(args)

        self._network = SPIEBaseNet(args, True)
        self.shared_cls_mean = dict()
        self.shared_cls_cov = dict()
        self.task_class_ranges = []

        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.min_lr = args["min_lr"]
        self.args = args
        self.share_lora_weight_decay = float(args["share_lora_weight_decay"])
        self.expert_head_weight_decay = float(args["expert_head_weight_decay"])

        self.task0_shared_epochs = int(args["task0_shared_epochs"])
        self.task0_shared_lr = float(args["task0_shared_lr"])
        self.shared_cls_epochs = int(args["shared_cls_epochs"])
        self.shared_cls_lr = float(args["shared_cls_lr"])
        self.shared_cls_weight_decay = float(args["shared_cls_weight_decay"])
        self.shared_cls_ca_lr = float(args["shared_cls_ca_lr"])
        self.shared_cls_crct_epochs = int(args["shared_cls_crct_epochs"])
        self.freeze_shared_lora_after_task0 = bool(args["freeze_shared_lora_after_task0"])

        self.task0_expert_epochs = int(args["task0_expert_epochs"])
        self.task0_expert_lr = float(args["task0_expert_lr"])
        self.incremental_expert_epochs = int(args["incremental_expert_epochs"])
        self.incremental_expert_lr = float(args["incremental_expert_lr"])

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
            "SPiE shared branch: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, freeze_shared_lora_after_task0=%s.",
            self.task0_shared_epochs,
            self.task0_shared_lr,
            self.shared_cls_epochs,
            self.shared_cls_lr,
            self.freeze_shared_lora_after_task0,
        )
        logging.info(
            (
                "SPiE expert branch: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, "
                "head=shared_tunalinear."
            ),
            self.task0_expert_epochs,
            self.task0_expert_lr,
            self.incremental_expert_epochs,
            self.incremental_expert_lr,
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

        use_backbone_dataparallel = bool(self.args["spie_backbone_dataparallel"])
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

    def _set_expert_head_requires_grad(self, task_id, requires_grad):
        self._network.get_expert_head(task_id).requires_grad_(requires_grad)

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

        base_jitter = float(self.args["covariance_regularization"])
        max_retry_power = int(self.args["max_covariance_retry_power"])
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
                optimizer.step()

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
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            expert_head.train()
            losses = 0.0
            ce_losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes
                expert_features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["expert_features"]
                expert_out = expert_head(expert_features)
                logits = expert_out["logits"]
                ce_loss = loss_cos(logits, local_targets)
                loss = ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                ce_losses += ce_loss.item()
                preds = torch.argmax(logits, dim=1) + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            avg_ce_loss = ce_losses / len(train_loader)
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
            )
            prog_bar.set_description(info)

        logging.info(info)

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

    def _centered_cosine_batch_np(self, shared_slices, expert_slices):
        shared_centered = shared_slices - shared_slices.mean(axis=1, keepdims=True)
        expert_centered = expert_slices - expert_slices.mean(axis=1, keepdims=True)
        numerator = np.sum(shared_centered * expert_centered, axis=1)
        denominator = np.linalg.norm(shared_centered, axis=1) * np.linalg.norm(expert_centered, axis=1)
        return numerator / np.clip(denominator, a_min=1e-12, a_max=None)

    def _zscore_rows_np(self, values):
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        return (values - mean) / np.clip(std, a_min=1e-12, a_max=None)

    def _build_class_to_task_np(self):
        class_to_task = np.full((self._total_classes,), -1, dtype=np.int64)
        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            class_to_task[start_idx:end_idx] = task_id
        return class_to_task

    def _candidate_tasks_from_topk_np(self, shared_topk, class_to_task, topk_width):
        candidate_tasks = []
        width = min(topk_width, shared_topk.shape[1])
        for row in shared_topk[:, :width]:
            row_tasks = []
            seen = set()
            for class_idx in row.tolist():
                task_id = int(class_to_task[class_idx])
                if task_id not in seen:
                    seen.add(task_id)
                    row_tasks.append(task_id)
            candidate_tasks.append(np.array(row_tasks, dtype=np.int64))
        return candidate_tasks

    def _filter_candidate_tasks_np(self, candidate_tasks, block_row, mode):
        if candidate_tasks.size <= 2:
            return candidate_tasks
        order = candidate_tasks[np.argsort(-block_row[candidate_tasks])]
        if mode == "top2_tasks_only":
            return order[:2]
        if mode == "top3_tasks_only":
            return order[:3]
        return candidate_tasks

    def _fixed_fusion_eval_np(self, shared_logits, expert_logits_by_task):
        params = {
            "alpha": 0.2,
            "beta_local": 0.02,
            "candidate_topk": 3,
            "candidate_task_mode": "top2_tasks_only",
            "gate_quantile": 0.1,
            "local_variant": "zscore",
            "multiply_mode": "additive",
            "apply_scope": "all_rerank_classes",
            "sim_variant": "center_cos",
            "rerank_topk": 3,
        }

        num_samples = shared_logits.shape[0]
        num_tasks = len(self.task_class_ranges)
        class_to_task = self._build_class_to_task_np()
        task_starts = np.array([start for start, _ in self.task_class_ranges], dtype=np.int64)
        task_ends = np.array([end for _, end in self.task_class_ranges], dtype=np.int64)

        max_topk = min(max(self.topk, params["candidate_topk"], params["rerank_topk"]), shared_logits.shape[1])
        shared_topk = np.argsort(-shared_logits, axis=1)[:, :max_topk]
        predicts = np.full((num_samples, self.topk), -1, dtype=np.int64)
        predicts[:, :max_topk] = shared_topk[:, :max_topk]

        if shared_logits.shape[1] <= 1:
            return predicts

        top1_class_gap = shared_logits[np.arange(num_samples), shared_topk[:, 0]] - shared_logits[np.arange(num_samples), shared_topk[:, 1]]
        gate_threshold = float(np.quantile(top1_class_gap, params["gate_quantile"]))

        block_scores = np.zeros((num_samples, num_tasks), dtype=np.float32)
        task_metric = np.zeros((num_samples, num_tasks), dtype=np.float32)
        expert_local_metric = []
        for task_id, (start_idx, end_idx) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
            shared_slice = shared_logits[:, start_idx:end_idx]
            expert_slice = expert_logits_by_task[task_id]
            max_shared = np.max(shared_slice, axis=1, keepdims=True)
            block_scores[:, task_id] = np.squeeze(
                max_shared + np.log(np.sum(np.exp(shared_slice - max_shared), axis=1, keepdims=True) + 1e-12),
                axis=1,
            )
            task_metric[:, task_id] = self._centered_cosine_batch_np(shared_slice, expert_slice).astype(np.float32)
            expert_local_metric.append(self._zscore_rows_np(expert_slice).astype(np.float32))

        candidate_tasks_by_k = self._candidate_tasks_from_topk_np(shared_topk, class_to_task, params["candidate_topk"])
        rerank_width = min(params["rerank_topk"], shared_topk.shape[1])

        for sample_idx in range(num_samples):
            if float(top1_class_gap[sample_idx]) > gate_threshold:
                continue

            candidate_classes = shared_topk[sample_idx, :rerank_width]
            candidate_tasks = self._filter_candidate_tasks_np(
                candidate_tasks_by_k[sample_idx], block_scores[sample_idx], params["candidate_task_mode"]
            )
            if candidate_tasks.size == 0:
                continue

            scores = shared_logits[sample_idx, candidate_classes].copy()
            candidate_task_set = set(candidate_tasks.tolist())
            for local_rank, class_idx in enumerate(candidate_classes.tolist()):
                task_id = int(class_to_task[class_idx])
                if task_id not in candidate_task_set:
                    continue
                local_idx = int(class_idx - task_starts[task_id])
                task_bonus = float(task_metric[sample_idx, task_id])
                local_bonus = float(expert_local_metric[task_id][sample_idx, local_idx])
                scores[local_rank] += params["alpha"] * task_bonus + params["beta_local"] * local_bonus

            reranked_classes = candidate_classes[np.argsort(-scores)]
            predicts[sample_idx, :rerank_width] = reranked_classes

        return predicts

    def eval_task(self):
        raw_cnn_pred, y_true = self._eval_cnn(self.test_loader)
        raw_cnn_accy = self._evaluate(raw_cnn_pred, y_true)

        raw_nme_pred, nme_true = self._eval_nme(self.test_loader, class_means=None)
        raw_nme_accy = self._evaluate(raw_nme_pred, nme_true)

        # SPiE reports the fusion branch as CNN and the shared-logit branch as NME.
        cnn_accy = raw_nme_accy
        nme_accy = raw_cnn_accy

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
        all_shared_logits, all_targets = [], []
        num_tasks = len(self.task_class_ranges)
        expert_logits_chunks = [[] for _ in range(num_tasks)]

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                shared_logits = self._shared_cls_logits(inputs)
                expert_logits_map = self._collect_expert_logits(inputs, list(range(num_tasks)))

            all_shared_logits.append(shared_logits.cpu().numpy().astype(np.float32))
            all_targets.append(targets.numpy())
            for task_id in range(num_tasks):
                expert_logits_chunks[task_id].append(expert_logits_map[task_id].cpu().numpy().astype(np.float32))

        shared_logits_np = np.concatenate(all_shared_logits, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        expert_logits_by_task = [np.concatenate(task_chunks, axis=0) for task_chunks in expert_logits_chunks]

        logging.info(
            "SPiE fusion eval branch (reported as CNN) uses fixed fusion: alpha=0.2 beta_local=0.02 candidate_topk=3 "
            "candidate_task_mode=top2_tasks_only gate_quantile=0.1 local_variant=zscore "
            "multiply_mode=additive apply_scope=all_rerank_classes sim_variant=center_cos rerank_topk=3."
        )
        y_pred = self._fixed_fusion_eval_np(shared_logits_np, expert_logits_by_task)
        return y_pred, y_true

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
