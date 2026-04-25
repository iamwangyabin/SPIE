import copy
import logging
import math

import numpy as np
import torch

from models.spie_base import Learner as SPIEBaseLearner


class Learner(SPIEBaseLearner):
    """SPiE keeps the shared/expert heads, then adds posterior fusion."""

    _spie_version_name = "SPiE"

    def __init__(self, args):
        super().__init__(args)

        self.posterior_task_temperature = float(args.get("posterior_task_temperature", 1.0))
        self.posterior_expert_temperature = float(args.get("posterior_expert_temperature", 1.0))
        self.posterior_shared_temperature = float(args.get("posterior_shared_temperature", 1.0))
        self.posterior_alpha = float(args.get("posterior_alpha", 0.2))
        self.posterior_alpha = min(max(self.posterior_alpha, 0.0), 1.0)
        self._eval_variants = {}

        logging.info(
            (
                "SPiE expert branch trains without shared-logit distillation. "
                "Posterior fusion: alpha=%s task_temperature=%s expert_temperature=%s shared_temperature=%s."
            ),
            self.posterior_alpha,
            self.posterior_task_temperature,
            self.posterior_expert_temperature,
            self.posterior_shared_temperature,
        )

    def consume_eval_variants(self):
        payload = copy.deepcopy(self._eval_variants) if self._eval_variants else None
        self._eval_variants = {}
        return payload

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
        else:
            self._train_shared_branch(
                train_loader=train_loader,
                epochs=self.shared_cls_epochs,
                branch_lr=self.shared_cls_lr,
                stage="shared_branch",
            )

        self._set_shared_lora_requires_grad(False)
        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

        if self._cur_task == 0:
            self._train_current_expert(
                train_loader=train_loader,
                epochs=self.task0_expert_epochs,
                expert_lr=self.task0_expert_lr,
                stage="task0_expert_local",
            )
        else:
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

    def _stable_softmax_np(self, logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)

    def _predict_topk_np(self, scores):
        topk = min(self.topk, scores.shape[1])
        predicts = np.full((scores.shape[0], self.topk), -1, dtype=np.int64)
        predicts[:, :topk] = np.argsort(-scores, axis=1)[:, :topk]
        return predicts

    def _collect_eval_logits_np(self, loader):
        self._network.eval()
        all_shared_logits, all_targets = [], []
        num_tasks = len(self.task_class_ranges)
        expert_logits_chunks = [[] for _ in range(num_tasks)]

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                shared_logits = self._shared_cls_logits(inputs)
                expert_logits_map = self._collect_expert_logits(inputs, list(range(num_tasks))) if num_tasks > 0 else {}

            all_shared_logits.append(shared_logits.cpu().numpy().astype(np.float32))
            all_targets.append(targets.numpy())
            for task_id in range(num_tasks):
                expert_logits_chunks[task_id].append(expert_logits_map[task_id].cpu().numpy().astype(np.float32))

        shared_logits_np = np.concatenate(all_shared_logits, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        expert_logits_by_task = [
            np.concatenate(task_chunks, axis=0) if task_chunks else np.zeros((shared_logits_np.shape[0], 0), dtype=np.float32)
            for task_chunks in expert_logits_chunks
        ]
        return shared_logits_np, expert_logits_by_task, y_true

    def _posterior_fusion_probs_np(self, shared_logits, expert_logits_by_task):
        shared_temperature = max(self.posterior_shared_temperature, 1e-6)
        task_temperature = max(self.posterior_task_temperature, 1e-6)
        expert_temperature = max(self.posterior_expert_temperature, 1e-6)

        p_shared = self._stable_softmax_np(shared_logits / shared_temperature).astype(np.float32)

        if not self.task_class_ranges:
            return p_shared, p_shared

        num_samples = shared_logits.shape[0]
        num_tasks = len(self.task_class_ranges)
        task_logits = np.zeros((num_samples, num_tasks), dtype=np.float32)
        p_moe = np.zeros_like(p_shared, dtype=np.float32)

        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            task_width = max(end_idx - start_idx, 1)
            shared_slice = shared_logits[:, start_idx:end_idx] / task_temperature
            max_slice = np.max(shared_slice, axis=1, keepdims=True)
            task_logits[:, task_id] = np.squeeze(
                max_slice + np.log(np.sum(np.exp(shared_slice - max_slice), axis=1, keepdims=True) + 1e-12),
                axis=1,
            ) - math.log(task_width)

        p_task = self._stable_softmax_np(task_logits).astype(np.float32)

        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            expert_logits = expert_logits_by_task[task_id]
            if expert_logits.shape[1] == 0:
                continue
            local_prob = self._stable_softmax_np(expert_logits / expert_temperature).astype(np.float32)
            p_moe[:, start_idx:end_idx] = p_task[:, task_id : task_id + 1] * local_prob

        p_final = self.posterior_alpha * p_shared + (1.0 - self.posterior_alpha) * p_moe
        return p_moe, p_final

    def eval_task(self):
        shared_logits_np, expert_logits_by_task, y_true = self._collect_eval_logits_np(self.test_loader)

        shared_pred = self._predict_topk_np(shared_logits_np)
        p_moe, p_final = self._posterior_fusion_probs_np(shared_logits_np, expert_logits_by_task)
        p_moe_pred = self._predict_topk_np(p_moe)
        p_final_pred = self._predict_topk_np(p_final)

        shared_accy = self._evaluate(shared_pred, y_true)
        p_moe_accy = self._evaluate(p_moe_pred, y_true)
        p_final_accy = self._evaluate(p_final_pred, y_true)

        self._eval_variants = {
            "shared_fc": shared_accy,
            "p_moe": p_moe_accy,
            "p_final": p_final_accy,
        }

        logging.info(
            (
                "SPiE eval variants: shared_fc uses shared logits top-k; "
                "p_moe uses p(task|x) * p(class|x,task); p_final mixes shared posterior and p_moe with alpha=%s."
            ),
            self.posterior_alpha,
        )

        return p_final_accy, p_moe_accy
