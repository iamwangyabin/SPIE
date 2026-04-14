import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import CosineLinear
from models.spie_v13 import Learner as SPiEV13Learner
from models.tuna import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy


class ExpertPrototypeBank(nn.Module):
    def __init__(self, prototypes=None, class_ids=None):
        super().__init__()
        prototypes = torch.zeros((0, 0), dtype=torch.float32) if prototypes is None else prototypes
        class_ids = torch.zeros((0,), dtype=torch.long) if class_ids is None else class_ids
        self.register_buffer("prototypes", prototypes)
        self.register_buffer("class_ids", class_ids)

    @property
    def is_empty(self):
        return self.prototypes.numel() == 0 or self.class_ids.numel() == 0


class Learner(SPiEV13Learner):
    """SPiE v15 learner with temporary per-expert cosine heads and prototype fusion inference."""

    _spie_version_name = "SPiE v15"

    def __init__(self, args):
        super().__init__(args)
        self.args["enable_expert_calibration"] = False
        self._network.enable_expert_calibration = False
        self.expert_calibration_epochs = 0

        self._network.expert_prototype_banks = nn.ModuleList()

        self.expert_filter_topk = int(args.get("expert_filter_topk", 0))
        self.expert_filter_threshold = float(args.get("expert_filter_threshold", 0.0))
        self.expert_weight_temperature = float(args.get("expert_weight_temperature", 1.0))
        self.expert_unavailable_score = float(args.get("expert_unavailable_score", -1e4))
        self.prototype_batch_size = int(args.get("prototype_batch_size", self.batch_size))
        self.prototype_augmentation_passes = int(args.get("prototype_augmentation_passes", 0))
        self.prototype_use_clean_pass = bool(args.get("prototype_use_clean_pass", True))
        self.prototype_aug_blend = float(args.get("prototype_aug_blend", 0.5))

        logging.info(
            "SPiE v15 prototype fusion: filter_topk=%s, filter_threshold=%s, weight_temperature=%s, unavailable_score=%s, prototype_aug_passes=%s, prototype_use_clean=%s, prototype_aug_blend=%s.",
            self.expert_filter_topk,
            self.expert_filter_threshold,
            self.expert_weight_temperature,
            self.expert_unavailable_score,
            self.prototype_augmentation_passes,
            self.prototype_use_clean_pass,
            self.prototype_aug_blend,
        )

    def _prepare_task_modules_for_load(self, task_id, data_manager, state_dict):
        del data_manager
        prototype_prefix = "expert_prototype_banks."
        bank_ids = set()
        for key in state_dict.keys():
            if key.startswith(prototype_prefix):
                try:
                    bank_ids.add(int(key.split(".")[1]))
                except (IndexError, ValueError):
                    continue

        target_banks = max(bank_ids) + 1 if bank_ids else 0
        while len(self._network.expert_prototype_banks) < target_banks:
            self._network.expert_prototype_banks.append(ExpertPrototypeBank())

    def _make_local_cosine_head(self, out_dim):
        head = CosineLinear(self._network.feature_dim, out_dim).to(self._device)
        return head

    def _run_local_head_training(
        self,
        train_loader,
        adapter_id,
        epochs,
        stage,
        head_lr,
        module_lr,
        module_weight_decay,
        toggle_modules,
    ):
        if epochs <= 0:
            return

        local_classes = self._total_classes - self._known_classes
        head = self._make_local_cosine_head(local_classes)
        toggle_modules()

        optimizer = self._make_optimizer(
            [
                {
                    "params": [p for p in self._backbone_module().parameters() if p.requires_grad],
                    "lr": module_lr,
                    "weight_decay": module_weight_decay,
                },
                {
                    "params": head.parameters(),
                    "lr": head_lr,
                    "weight_decay": self.weight_decay,
                },
            ]
        )
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            head.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                local_targets = targets - self._known_classes

                features = self._network.backbone(inputs, adapter_id=adapter_id, train=True)["features"]
                logits = head(features)["logits"]
                loss = loss_cos(logits, local_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq(local_targets).cpu().sum()
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

    def _train_task0_shared_lora(self, train_loader):
        self._run_local_head_training(
            train_loader=train_loader,
            adapter_id=-1,
            epochs=self.task0_shared_epochs,
            stage="task0_shared_lora",
            head_lr=self.init_lr,
            module_lr=self.task0_shared_lr,
            module_weight_decay=self.share_lora_weight_decay,
            toggle_modules=lambda: (
                self._set_shared_lora_requires_grad(True),
                self._set_current_expert_requires_grad(False),
            ),
        )

    def _train_task0_expert(self, train_loader):
        self._run_local_head_training(
            train_loader=train_loader,
            adapter_id=self._cur_task,
            epochs=self.task0_expert_epochs,
            stage="task0_expert",
            head_lr=self.init_lr,
            module_lr=self.task0_expert_lr,
            module_weight_decay=self.weight_decay,
            toggle_modules=lambda: (
                self._set_shared_lora_requires_grad(False),
                self._set_current_expert_requires_grad(True),
            ),
        )
        self._set_shared_lora_requires_grad(False)

    def _train_incremental_expert(self, train_loader):
        self._run_local_head_training(
            train_loader=train_loader,
            adapter_id=self._cur_task,
            epochs=self.incremental_expert_epochs,
            stage="incremental_expert",
            head_lr=self.init_lr,
            module_lr=self.incremental_expert_lr,
            module_weight_decay=self.weight_decay,
            toggle_modules=lambda: (
                self._freeze_shared_domain_adapter(),
                self._set_current_expert_requires_grad(True),
            ),
        )

    def _train_shared_delta(self, train_loader):
        if self.shared_update_epochs <= 0 or self.shared_ema_alpha <= 0.0:
            return

        backbone = self._backbone_module()
        original_shared_adapter = backbone.cur_shared_adapter
        try:
            shared_work = copy.deepcopy(original_shared_adapter).to(self._device)
            original_shared_adapter.requires_grad_(False)
            shared_work.requires_grad_(True)
            self._set_current_expert_requires_grad(False)
            backbone.cur_shared_adapter = shared_work

            self._run_local_head_training(
                train_loader=train_loader,
                adapter_id=-1,
                epochs=self.shared_update_epochs,
                stage="shared_delta",
                head_lr=self.init_lr,
                module_lr=self.shared_update_lr,
                module_weight_decay=self.share_lora_weight_decay,
                toggle_modules=lambda: None,
            )
            self._ema_update_shared_adapter(original_shared_adapter, shared_work)
        finally:
            backbone.cur_shared_adapter = original_shared_adapter
            self._freeze_shared_domain_adapter()

    def _train_expert_calibration(self, train_loader):
        del train_loader
        return

    def _ensure_prototype_bank(self, expert_id):
        while len(self._network.expert_prototype_banks) <= expert_id:
            self._network.expert_prototype_banks.append(ExpertPrototypeBank())

    @torch.no_grad()
    def _extract_class_feature_mean(self, class_idx, mode):
        _, _, idx_dataset = self.data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1),
            source="train",
            mode=mode,
            ret_data=True,
        )
        idx_loader = DataLoader(
            idx_dataset,
            batch_size=self.prototype_batch_size,
            shuffle=False,
            num_workers=4,
        )

        vectors = []
        for _, _inputs, _targets in idx_loader:
            del _targets
            features = self._network.backbone(_inputs.to(self._device), adapter_id=self._cur_task, train=False)["features"]
            vectors.append(F.normalize(features, p=2, dim=1))

        return F.normalize(torch.cat(vectors, dim=0).mean(dim=0, keepdim=True), p=2, dim=1)

    @torch.no_grad()
    def _compute_current_expert_prototypes(self, model):
        model.eval()
        class_ids = list(range(self._known_classes, self._total_classes))
        prototypes = []

        for class_idx in class_ids:
            clean_proto = None
            if self.prototype_use_clean_pass:
                clean_proto = self._extract_class_feature_mean(class_idx, mode="test")

            aug_proto = None
            if self.prototype_augmentation_passes > 0:
                aug_vectors = []
                for _ in range(self.prototype_augmentation_passes):
                    aug_vectors.append(self._extract_class_feature_mean(class_idx, mode="train"))
                aug_proto = F.normalize(torch.cat(aug_vectors, dim=0).mean(dim=0, keepdim=True), p=2, dim=1)

            if clean_proto is not None and aug_proto is not None:
                blend = min(max(self.prototype_aug_blend, 0.0), 1.0)
                proto = F.normalize((1.0 - blend) * clean_proto + blend * aug_proto, p=2, dim=1)
            elif aug_proto is not None:
                proto = aug_proto
            elif clean_proto is not None:
                proto = clean_proto
            else:
                proto = self._extract_class_feature_mean(class_idx, mode="test")

            prototypes.append(proto)

        prototype_tensor = torch.cat(prototypes, dim=0)
        class_id_tensor = torch.tensor(class_ids, device=prototype_tensor.device, dtype=torch.long)
        self._ensure_prototype_bank(self._cur_task)
        self._network.expert_prototype_banks[self._cur_task] = ExpertPrototypeBank(
            prototypes=prototype_tensor.detach().clone(),
            class_ids=class_id_tensor,
        ).to(prototype_tensor.device)

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
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
        self._compute_current_expert_prototypes(backbone)

        self._train_shared_cls_classifier(train_loader)
        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

    def _shared_cls_probabilities(self, inputs):
        if self._network.fc_shared_cls is None:
            return None
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        shared_logits = self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]
        return F.softmax(self.args["scale"] * shared_logits, dim=1)

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

    def _stack_expert_prototype_scores(self, inputs, active_expert_ids):
        global_scores = []
        expert_confidences = []

        for expert_id in active_expert_ids:
            bank = self._network.expert_prototype_banks[expert_id]
            if bank.is_empty:
                continue

            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
            features = F.normalize(features, p=2, dim=1)
            prototypes = F.normalize(bank.prototypes.to(features.device, dtype=features.dtype), p=2, dim=1)
            local_scores = torch.matmul(features, prototypes.T)
            expert_confidences.append(local_scores.max(dim=1).values)

            scores = local_scores.new_full((local_scores.shape[0], self._total_classes), self.expert_unavailable_score)
            class_ids = bank.class_ids.to(device=features.device, dtype=torch.long)
            scores.index_copy_(1, class_ids, local_scores)
            global_scores.append(scores)

        if not global_scores:
            empty_scores = torch.full(
                (0, inputs.shape[0], self._total_classes),
                self.expert_unavailable_score,
                device=inputs.device,
            )
            empty_conf = torch.zeros((0, inputs.shape[0]), device=inputs.device)
            return empty_scores, empty_conf

        return torch.stack(global_scores, dim=0), torch.stack(expert_confidences, dim=0)

    def _filter_and_fuse_expert_scores(self, stacked_scores, expert_confidences):
        if stacked_scores.shape[0] == 0:
            return torch.full(
                (stacked_scores.shape[1], stacked_scores.shape[2]),
                self.expert_unavailable_score,
                device=stacked_scores.device,
            )

        confidence_mask = torch.ones_like(expert_confidences, dtype=torch.bool)
        if self.expert_filter_threshold > 0.0:
            best_confidence = expert_confidences.max(dim=0, keepdim=True).values
            confidence_mask &= expert_confidences >= (best_confidence - self.expert_filter_threshold)

        if self.expert_filter_topk > 0 and self.expert_filter_topk < stacked_scores.shape[0]:
            topk_indices = expert_confidences.topk(k=self.expert_filter_topk, dim=0).indices
            topk_mask = torch.zeros_like(confidence_mask)
            topk_mask.scatter_(0, topk_indices, True)
            confidence_mask &= topk_mask

        masked_confidence = expert_confidences.masked_fill(~confidence_mask, float("-inf"))
        all_filtered = ~confidence_mask.any(dim=0)
        if all_filtered.any():
            best_indices = expert_confidences.argmax(dim=0, keepdim=True)
            confidence_mask.scatter_(0, best_indices, True)
            masked_confidence = expert_confidences.masked_fill(~confidence_mask, float("-inf"))

        temperature = max(self.expert_weight_temperature, 1e-8)
        expert_weights = F.softmax(masked_confidence / temperature, dim=0)
        expert_weights = expert_weights.masked_fill(~confidence_mask, 0.0)

        unavailable_mask = stacked_scores <= (self.expert_unavailable_score + 1.0)
        safe_scores = stacked_scores.masked_fill(unavailable_mask, 0.0)
        fused_scores = (expert_weights.unsqueeze(-1) * safe_scores).sum(dim=0)
        has_support = (~unavailable_mask & confidence_mask.unsqueeze(-1)).any(dim=0)
        fused_scores = fused_scores.masked_fill(~has_support, self.expert_unavailable_score)
        return fused_scores

    def _eval_expert_prototypes(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_expert_ids = [idx for idx in self._active_expert_ids() if idx < len(self._network.expert_prototype_banks)]

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                stacked_scores, expert_confidences = self._stack_expert_prototype_scores(inputs, active_expert_ids)
                scores = self._filter_and_fuse_expert_scores(stacked_scores, expert_confidences)

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

    def eval_task(self):
        y_pred, y_true = self._eval_shared_cls(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        y_pred, y_true = self._eval_expert_prototypes(self.test_loader)
        nme_accy = self._evaluate(y_pred, y_true)

        return cnn_accy, nme_accy
