import logging
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.spie_v13 import Learner as SPiEV13Learner


class AnalyticalExpertHead(nn.Module):
    def __init__(
        self,
        in_features,
        class_offset,
        gamma=500.0,
        normalize_input=True,
        buffer_size=0,
        use_relu_buffer=True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.gamma = float(gamma)
        self.normalize_input = bool(normalize_input)
        self.buffer_size = int(buffer_size)
        self.use_relu_buffer = bool(use_relu_buffer)
        self.projected_dim = self.buffer_size if self.buffer_size > 0 else self.in_features

        self.register_buffer("class_offset", torch.tensor(int(class_offset), dtype=torch.long))
        if self.buffer_size > 0:
            projection = torch.empty((self.in_features, self.buffer_size), dtype=torch.double)
            nn.init.kaiming_uniform_(projection, a=math.sqrt(5.0))
            self.register_buffer("projection", projection)
        else:
            self.register_buffer("projection", torch.zeros((0, 0), dtype=torch.double))
        self.register_buffer("weight", torch.zeros((self.projected_dim, 0), dtype=torch.double))
        self.register_buffer(
            "R",
            torch.eye(self.projected_dim, dtype=torch.double) / max(self.gamma, 1e-8),
        )

    @property
    def output_dim(self):
        return int(self.weight.shape[1])

    def _prepare_features(self, features):
        if self.normalize_input:
            features = F.normalize(features, p=2, dim=1)
        features = features.to(dtype=self.weight.dtype, device=self.weight.device)
        if self.buffer_size > 0:
            features = features @ self.projection
            if self.use_relu_buffer:
                features = F.relu(features)
        return features

    def expand_output_dim(self, num_classes):
        num_classes = int(num_classes)
        if num_classes <= self.output_dim:
            return
        tail = torch.zeros(
            (self.projected_dim, num_classes - self.output_dim),
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
        self.args["enable_expert_calibration"] = False
        self._network.enable_expert_calibration = False
        self.expert_head_gamma = float(args.get("expert_head_gamma", 500.0))
        self.expert_head_fit_epochs = int(args.get("expert_head_fit_epochs", args.get("fit_epochs", 3)))
        self.expert_head_normalize_input = bool(args.get("expert_head_normalize_input", True))
        self.expert_head_buffer_size = int(args.get("expert_head_buffer_size", 0))
        self.expert_head_use_relu_buffer = bool(args.get("expert_head_use_relu_buffer", True))
        self.expert_unavailable_logit = float(args.get("expert_unavailable_logit", -1e4))
        self.expert_calibration_epochs = 0

        self._network.expert_analytical_heads = nn.ModuleList()
        logging.info(
            "SPiE v14 analytical expert heads: fit_epochs=%s, gamma=%s, normalize_input=%s, buffer_size=%s, relu_buffer=%s.",
            self.expert_head_fit_epochs,
            self.expert_head_gamma,
            self.expert_head_normalize_input,
            self.expert_head_buffer_size,
            self.expert_head_use_relu_buffer,
        )
        logging.info("SPiE v14 disables expert calibration and uses raw analytical expert logits at evaluation.")

    def _append_analytical_head(self, class_offset):
        backbone_device = next(self._network.backbone.parameters()).device
        head = AnalyticalExpertHead(
            in_features=self._network.feature_dim,
            class_offset=class_offset,
            gamma=self.expert_head_gamma,
            normalize_input=self.expert_head_normalize_input,
            buffer_size=self.expert_head_buffer_size,
            use_relu_buffer=self.expert_head_use_relu_buffer,
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

    def _refit_analytical_expert_heads(self, train_loader):
        active_expert_ids = self._active_expert_ids()
        if not active_expert_ids:
            return

        self._ensure_current_analytical_head()
        self._network.backbone.eval()
        total_batches = max(len(train_loader), 1)

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
                "Task %d --> Refit analytical expert heads (%d/%d)",
                self._cur_task,
                batch_idx,
                total_batches,
            )

        self._record_extra_stage_epoch(
            stage="refit_analytical_expert_heads",
            epoch=1,
            total_epochs=1,
            num_experts=len(active_expert_ids),
            batches=total_batches,
        )

    def _build_global_expert_scores(self, local_scores, head, fill_value):
        batch_size = local_scores.shape[0]
        scores = local_scores.new_full((batch_size, self._total_classes), fill_value)
        class_offset = int(head.class_offset.item())
        if class_offset >= self._total_classes or local_scores.shape[1] == 0:
            return scores

        usable_classes = min(local_scores.shape[1], self._total_classes - class_offset)
        scores[:, class_offset : class_offset + usable_classes] = local_scores[:, :usable_classes]
        return scores

    def _stack_expert_probabilities(self, inputs, active_expert_ids):
        probs_per_expert = []
        for expert_id in active_expert_ids:
            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
            head = self._network.expert_analytical_heads[expert_id]
            local_logits = head(features)
            local_probs = F.softmax(local_logits, dim=1)
            probs = self._build_global_expert_scores(local_probs, head, fill_value=0.0)
            probs_per_expert.append(probs)
        return torch.stack(probs_per_expert, dim=0)

    def _best_expert_probabilities(self, stacked_probs):
        return stacked_probs.max(dim=0).values

    def _shared_cls_probabilities(self, inputs):
        if self._network.fc_shared_cls is None:
            return None
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        shared_logits = self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]
        return F.softmax(self.args["scale"] * shared_logits, dim=1)

    def _train_expert_calibration(self, train_loader):
        del train_loader
        return

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

        self._fit_analytical_expert_heads(train_loader)
        self._refit_analytical_expert_heads(self.train_loader_for_protonet)
        self._train_expert_calibration(self.train_loader_for_protonet)

        self._train_shared_cls_classifier(train_loader)
        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if self._network.fc_shared_cls is not None:
            y_pred, y_true = self._eval_shared_cls(self.test_loader)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_expert_ids = self._active_expert_ids()

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                stacked_probs = self._stack_expert_probabilities(inputs, active_expert_ids)
                scores = self._best_expert_probabilities(stacked_probs)

            topk = min(self.topk, scores.shape[1])
            predicts = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)[1]
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

    def _eval_shared_cls(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                scores = self._shared_cls_probabilities(inputs)

            topk = min(self.topk, scores.shape[1])
            predicts = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)[1]
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
