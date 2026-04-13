import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from models.spie_v13 import Learner as SPiEV13Learner
from utils.toolkit import tensor2numpy


class AnalyticalExpertHead(nn.Module):
    def __init__(self, in_features, class_offset, gamma=500.0, normalize_input=True):
        super().__init__()
        self.in_features = int(in_features)
        self.gamma = float(gamma)
        self.normalize_input = bool(normalize_input)

        self.register_buffer("class_offset", torch.tensor(int(class_offset), dtype=torch.long))
        self.register_buffer("weight", torch.zeros((self.in_features, 0), dtype=torch.double))
        self.register_buffer(
            "R",
            torch.eye(self.in_features, dtype=torch.double) / max(self.gamma, 1e-8),
        )

    @property
    def output_dim(self):
        return int(self.weight.shape[1])

    def _prepare_features(self, features):
        if self.normalize_input:
            features = F.normalize(features, p=2, dim=1)
        return features.to(dtype=self.weight.dtype, device=self.weight.device)

    def expand_output_dim(self, num_classes):
        num_classes = int(num_classes)
        if num_classes <= self.output_dim:
            return
        tail = torch.zeros(
            (self.in_features, num_classes - self.output_dim),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )
        self.weight = torch.cat((self.weight, tail), dim=1)

    @torch.no_grad()
    def fit_batch(self, features, global_targets, total_classes):
        total_classes = int(total_classes)
        class_offset = int(self.class_offset.item())
        local_classes = max(total_classes - class_offset, 0)
        if local_classes <= 0:
            return

        self.expand_output_dim(local_classes)
        features = self._prepare_features(features)
        local_targets = global_targets.to(device=features.device, dtype=torch.long) - class_offset
        if torch.any(local_targets < 0) or torch.any(local_targets >= local_classes):
            raise ValueError(
                f"Analytical expert head offset={class_offset} received targets outside "
                f"[{class_offset}, {total_classes})."
            )

        one_hot = F.one_hot(local_targets, num_classes=local_classes).to(dtype=self.weight.dtype)
        k = torch.inverse(torch.eye(features.shape[0], device=features.device, dtype=features.dtype) + features @ self.R @ features.T)
        self.R -= self.R @ features.T @ k @ features @ self.R
        self.weight += self.R @ features.T @ (one_hot - features @ self.weight)

    def forward(self, features):
        if self.output_dim == 0:
            return features.new_zeros((features.shape[0], 0))
        logits = self._prepare_features(features) @ self.weight
        return logits.to(dtype=features.dtype)


class Learner(SPiEV13Learner):
    """SPiE v14 learner with analytical per-expert classifiers."""

    _spie_version_name = "SPiE v14"

    def __init__(self, args):
        super().__init__(args)
        self.expert_head_gamma = float(args.get("expert_head_gamma", 500.0))
        self.expert_head_fit_epochs = int(args.get("expert_head_fit_epochs", args.get("fit_epochs", 3)))
        self.expert_head_normalize_input = bool(args.get("expert_head_normalize_input", True))
        self.expert_unavailable_logit = float(args.get("expert_unavailable_logit", -1e4))

        self._network.expert_analytical_heads = nn.ModuleList()
        logging.info(
            "SPiE v14 analytical expert heads: fit_epochs=%s, gamma=%s, normalize_input=%s.",
            self.expert_head_fit_epochs,
            self.expert_head_gamma,
            self.expert_head_normalize_input,
        )

    def _append_analytical_head(self, class_offset):
        backbone_device = next(self._network.backbone.parameters()).device
        head = AnalyticalExpertHead(
            in_features=self._network.feature_dim,
            class_offset=class_offset,
            gamma=self.expert_head_gamma,
            normalize_input=self.expert_head_normalize_input,
        ).to(backbone_device)
        self._network.expert_analytical_heads.append(head)
        return head

    def _ensure_current_analytical_head(self):
        while len(self._network.expert_analytical_heads) <= self._cur_task:
            self._append_analytical_head(class_offset=self._known_classes)

    def _prepare_task_modules_for_load(self, task_id, data_manager, state_dict):
        current_heads = len(self._network.expert_analytical_heads)
        known_classes = sum(data_manager.get_task_size(mapped_task_id) for mapped_task_id in range(current_heads))
        for mapped_task_id in range(current_heads, task_id + 1):
            head = self._append_analytical_head(class_offset=known_classes)
            weight_key = f"expert_analytical_heads.{mapped_task_id}.weight"
            if weight_key in state_dict:
                head.expand_output_dim(int(state_dict[weight_key].shape[1]))
            known_classes += data_manager.get_task_size(mapped_task_id)

    def _fit_analytical_expert_heads(self, train_loader):
        if self.expert_head_fit_epochs <= 0:
            return

        active_expert_ids = self._active_expert_ids()
        if not active_expert_ids:
            return

        self._ensure_current_analytical_head()
        self._network.backbone.eval()
        total_batches = max(len(train_loader), 1)

        for epoch in range(self.expert_head_fit_epochs):
            for batch_idx, (_, inputs, targets) in enumerate(train_loader, start=1):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                with torch.no_grad():
                    for expert_id in active_expert_ids:
                        features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
                        self._network.expert_analytical_heads[expert_id].fit_batch(
                            features=features,
                            global_targets=targets,
                            total_classes=self._total_classes,
                        )

                logging.info(
                    "Task %d --> Fit analytical expert heads (%d/%d, batch %d/%d)",
                    self._cur_task,
                    epoch + 1,
                    self.expert_head_fit_epochs,
                    batch_idx,
                    total_batches,
                )

            self._record_extra_stage_epoch(
                stage="fit_analytical_expert_heads",
                epoch=epoch + 1,
                total_epochs=self.expert_head_fit_epochs,
                num_experts=len(active_expert_ids),
            )

    def _build_global_expert_logits(self, local_logits, head):
        batch_size = local_logits.shape[0]
        logits = local_logits.new_full((batch_size, self._total_classes), self.expert_unavailable_logit)
        class_offset = int(head.class_offset.item())
        if class_offset >= self._total_classes or local_logits.shape[1] == 0:
            return logits

        usable_classes = min(local_logits.shape[1], self._total_classes - class_offset)
        logits[:, class_offset : class_offset + usable_classes] = local_logits[:, :usable_classes]
        return logits

    def _stack_calibrated_expert_logits(self, inputs, active_expert_ids):
        logits_per_expert = []
        for expert_id in active_expert_ids:
            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
            head = self._network.expert_analytical_heads[expert_id]
            logits = self._build_global_expert_logits(head(features), head)
            logits_per_expert.append(self._network.calibrate_expert_logits(logits, expert_id))
        return torch.stack(logits_per_expert, dim=0)

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
        calibration_params = list(self._network.expert_calibration_log_scale) + list(self._network.expert_calibration_bias)
        backbone_grad_flags = [param.requires_grad for param in backbone_params]
        calibration_grad_flags = [param.requires_grad for param in calibration_params]

        try:
            self._network.backbone.eval()
            for param in backbone_params:
                param.requires_grad = False
            for param in calibration_params:
                param.requires_grad = True

            routing_temperature = max(self.expert_calibration_routing_temperature, 1e-8)
            for _, epoch in enumerate(prog_bar):
                losses = 0.0
                correct, total = 0, 0

                for _, (_, inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)

                    frozen_logits = []
                    with torch.no_grad():
                        for expert_id in active_expert_ids:
                            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
                            head = self._network.expert_analytical_heads[expert_id]
                            logits = self._build_global_expert_logits(head(features), head)
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
            self._restore_requires_grad(calibration_params, calibration_grad_flags)

        logging.info(info)

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
            if self.task0_copy_shared_to_expert and self.task0_shared_epochs > 0:
                self._copy_shared_lora_to_current_expert()
            self._train_task0_expert(train_loader)
        else:
            self._train_incremental_expert(train_loader)
            self._train_shared_delta(train_loader)

        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        backbone_module.adapter_update()

        self._fit_analytical_expert_heads(self.train_loader_for_protonet)
        self._train_expert_calibration(self.train_loader_for_protonet)

        self._train_shared_cls_classifier(train_loader)
        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()
