import logging

import numpy as np
import torch

from models.spie_v16 import Learner as SPIEV16Learner


class Learner(SPIEV16Learner):
    """SPiE v18 keeps v16 training but simplifies inference to shared score plus relation matching."""

    _spie_version_name = "SPiE v18"

    def _centered_cosine_batch_np(self, shared_slices, expert_slices):
        shared_centered = shared_slices - shared_slices.mean(axis=1, keepdims=True)
        expert_centered = expert_slices - expert_slices.mean(axis=1, keepdims=True)
        numerator = np.sum(shared_centered * expert_centered, axis=1)
        denominator = np.linalg.norm(shared_centered, axis=1) * np.linalg.norm(expert_centered, axis=1)
        return numerator / np.clip(denominator, a_min=1e-12, a_max=None)

    def _simple_relation_fusion_eval_np(self, shared_logits, expert_logits_by_task):
        topk = min(self.topk, shared_logits.shape[1])
        predicts = np.full((shared_logits.shape[0], self.topk), -1, dtype=np.int64)

        if not self.task_class_ranges:
            predicts[:, :topk] = np.argsort(-shared_logits, axis=1)[:, :topk]
            return predicts

        fused_logits = shared_logits.copy()
        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            shared_slice = shared_logits[:, start_idx:end_idx]
            expert_slice = expert_logits_by_task[task_id]
            relation_score = self._centered_cosine_batch_np(shared_slice, expert_slice).astype(np.float32)
            fused_logits[:, start_idx:end_idx] += relation_score[:, None]

        predicts[:, :topk] = np.argsort(-fused_logits, axis=1)[:, :topk]
        return predicts

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
            "SPiE v18 fusion eval branch (reported as CNN) uses simple fusion: fused_score = global_score + relation_matching_score."
        )
        y_pred = self._simple_relation_fusion_eval_np(shared_logits_np, expert_logits_by_task)
        return y_pred, y_true
