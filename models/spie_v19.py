import copy
import logging
import math

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.spie_v16 import Learner as SPIEV16Learner
from utils.toolkit import tensor2numpy


class Learner(SPIEV16Learner):
    """SPiE v19 keeps the v16 heads but aligns experts to shared-shape and uses posterior fusion."""

    _spie_version_name = "SPiE v19"

    def __init__(self, args):
        super().__init__(args)

        self.expert_shape_distill_lambda = float(args.get("expert_shape_distill_lambda", 0.1))
        self.expert_shape_distill_temperature = float(args.get("expert_shape_distill_temperature", 2.0))
        self.expert_shape_reg_cap_ratio = float(args.get("expert_shape_reg_cap_ratio", 0.25))
        self.posterior_task_temperature = float(args.get("posterior_task_temperature", 1.0))
        self.posterior_expert_temperature = float(args.get("posterior_expert_temperature", 1.0))
        self.posterior_shared_temperature = float(args.get("posterior_shared_temperature", 1.0))
        self.posterior_alpha = float(args.get("posterior_alpha", 0.2))
        self.posterior_alpha = min(max(self.posterior_alpha, 0.0), 1.0)
        self._eval_variants = {}

        logging.info(
            (
                "SPiE v19 expert shape distill: lambda=%s temperature=%s cap_ratio=%s. "
                "Posterior fusion: alpha=%s task_temperature=%s expert_temperature=%s shared_temperature=%s."
            ),
            self.expert_shape_distill_lambda,
            self.expert_shape_distill_temperature,
            self.expert_shape_reg_cap_ratio,
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

    def _shape_distillation_loss(self, student_logits, teacher_logits):
        if self.expert_shape_distill_lambda <= 0 or student_logits.shape[1] <= 1:
            return student_logits.new_zeros(())

        temperature = max(self.expert_shape_distill_temperature, 1e-6)
        student_shape = self._zscore_tensor(student_logits, dim=1)
        teacher_shape = self._zscore_tensor(teacher_logits, dim=1)
        teacher_prob = F.softmax(teacher_shape / temperature, dim=1)
        student_log_prob = F.log_softmax(student_shape / temperature, dim=1)
        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)

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
        loss_cos = self._shape_aware_cosface()

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            expert_head.train()
            losses = 0.0
            ce_losses = 0.0
            shape_losses = 0.0
            shape_penalties = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes

                with torch.no_grad():
                    self._network.backbone.eval()
                    self._network.fc_shared_cls.eval()
                    shared_local_logits = self._shared_cls_logits(inputs)[:, self._known_classes : self._total_classes]
                self._network.backbone.train()
                expert_head.train()

                expert_features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["expert_features"]
                expert_out = expert_head(expert_features)
                logits = expert_out["logits"]

                ce_loss = loss_cos(logits, local_targets)
                shape_loss = self._shape_distillation_loss(logits, shared_local_logits)
                shape_penalty = self.expert_shape_distill_lambda * shape_loss
                if self.expert_shape_reg_cap_ratio > 0:
                    shape_penalty = torch.minimum(
                        shape_penalty,
                        ce_loss.detach() * self.expert_shape_reg_cap_ratio,
                    )
                loss = ce_loss + shape_penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                ce_losses += ce_loss.item()
                shape_losses += shape_loss.item()
                shape_penalties += shape_penalty.item()
                preds = torch.argmax(logits, dim=1) + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            avg_ce_loss = ce_losses / len(train_loader)
            avg_shape_loss = shape_losses / len(train_loader)
            avg_shape_penalty = shape_penalties / len(train_loader)
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
                shape_loss=float(avg_shape_loss),
                shape_penalty=float(avg_shape_penalty),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _shape_aware_cosface(self):
        return self._make_cosface_loss()

    def _make_cosface_loss(self):
        from models.tuna import AngularPenaltySMLoss

        return AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

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
                "SPiE v19 eval variants: shared_fc uses shared logits top-k; "
                "p_moe uses p(task|x) * p(class|x,task); p_final mixes shared posterior and p_moe with alpha=%s."
            ),
            self.posterior_alpha,
        )

        return p_final_accy, p_moe_accy
