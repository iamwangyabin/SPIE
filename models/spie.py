import copy
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

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
        self.posterior_router = str(args.get("posterior_router", "prototype_activation"))
        supported_routers = {"task_logmeanexp", "prototype_activation", "task_max_proto"}
        if self.posterior_router not in supported_routers:
            raise ValueError(
                f"Unknown posterior_router: {self.posterior_router}. "
                f"Supported routers: {sorted(supported_routers)}"
            )
        self._eval_variants = {}

        logging.info(
            (
                "SPiE expert branch trains without shared-logit distillation. "
                "Posterior fusion: posterior_router=%s alpha=%s task_temperature=%s "
                "expert_temperature=%s shared_temperature=%s."
            ),
            self.posterior_router,
            self.posterior_alpha,
            self.posterior_task_temperature,
            self.posterior_expert_temperature,
            self.posterior_shared_temperature,
        )

    def consume_eval_variants(self):
        payload = copy.deepcopy(self._eval_variants) if self._eval_variants else None
        self._eval_variants = {}
        return payload

    def _drop_topk_metrics(self, accy):
        return {
            key: value
            for key, value in accy.items()
            if key == "top1" or not key.startswith("top")
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

    def _find_last_linear_weight(self, module):
        last_weight = None
        for submodule in module.modules():
            weight = getattr(submodule, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                last_weight = weight
        return last_weight

    def _extract_class_prototype_bank(self):
        """
        Extract discriminative class prototypes from the calibrated shared classifier.

        The calibrated shared classifier weights are treated as a class-prototype bank
        for prototype activation routing.
        """
        fc = self._network.fc_shared_cls

        if hasattr(fc, "heads"):
            weights = []
            for head in fc.heads:
                weight = self._find_last_linear_weight(head)
                if weight is None:
                    raise RuntimeError(f"Cannot find Linear weight in shared classifier head: {head}")
                weights.append(weight)
            class_prototype_bank = torch.cat(weights, dim=0)
        else:
            class_prototype_bank = getattr(fc, "weight", None)
            if class_prototype_bank is None:
                raise RuntimeError(f"Unsupported fc_shared_cls type for prototype routing: {type(fc)}")

        class_prototype_bank = class_prototype_bank[: self._total_classes]
        return F.normalize(class_prototype_bank, p=2, dim=1)

    def _compute_prototype_activation(self, shared_features, class_prototype_bank):
        """Compute activation scores between shared features and class prototypes."""
        shared_features = F.normalize(shared_features, p=2, dim=1)
        class_prototype_bank = F.normalize(class_prototype_bank, p=2, dim=1)
        return shared_features @ class_prototype_bank.T

    def _class_to_task(self, pred_class):
        """Map global class indices to task indices according to self.task_class_ranges."""
        pred_task = torch.full_like(pred_class, fill_value=-1)

        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            mask = (pred_class >= start_idx) & (pred_class < end_idx)
            pred_task[mask] = task_id

        if torch.any(pred_task < 0):
            bad = pred_class[pred_task < 0][:10].detach().cpu().tolist()
            raise ValueError(f"Some classes are outside task ranges: {bad}")

        return pred_task

    def _task_posterior_from_prototype_activation(self, shared_features):
        """Route each sample to the task of its most activated class prototype."""
        class_prototype_bank = self._extract_class_prototype_bank()
        class_activation = self._compute_prototype_activation(shared_features, class_prototype_bank)
        pred_class = torch.argmax(class_activation, dim=1)
        pred_task = self._class_to_task(pred_class)

        task_route_prob = torch.zeros(
            shared_features.size(0),
            len(self.task_class_ranges),
            device=shared_features.device,
            dtype=shared_features.dtype,
        )
        task_route_prob[
            torch.arange(shared_features.size(0), device=shared_features.device),
            pred_task,
        ] = 1.0
        return task_route_prob

    def _task_posterior_from_task_max_prototype(self, shared_features):
        """Each task uses its most activated class prototype as task route score."""
        task_temperature = max(self.posterior_task_temperature, 1e-6)
        class_prototype_bank = self._extract_class_prototype_bank()
        class_activation = self._compute_prototype_activation(shared_features, class_prototype_bank)

        task_route_scores = []
        for start_idx, end_idx in self.task_class_ranges:
            block = class_activation[:, start_idx:end_idx] / task_temperature
            if block.shape[1] == 0:
                score = torch.full(
                    (shared_features.size(0),),
                    -torch.inf,
                    device=shared_features.device,
                    dtype=shared_features.dtype,
                )
            else:
                score = torch.max(block, dim=1).values
            task_route_scores.append(score)

        task_route_scores = torch.stack(task_route_scores, dim=1)
        return torch.softmax(task_route_scores, dim=1)

    def _task_posterior_from_task_logmeanexp(self, global_class_logits):
        """
        Original SPiE task router: aggregate class logits within each task by log-mean-exp.
        """
        task_temperature = max(self.posterior_task_temperature, 1e-6)
        task_route_scores = []

        for start_idx, end_idx in self.task_class_ranges:
            block = global_class_logits[:, start_idx:end_idx] / task_temperature
            task_width = max(end_idx - start_idx, 1)
            score = torch.logsumexp(block, dim=1) - math.log(task_width)
            task_route_scores.append(score)

        task_route_scores = torch.stack(task_route_scores, dim=1)
        return torch.softmax(task_route_scores, dim=1)

    def _task_posterior_from_task_logmeanexp_np(self, global_class_logits):
        global_class_logits_t = torch.from_numpy(global_class_logits).to(self._device, dtype=torch.float32)
        return (
            self._task_posterior_from_task_logmeanexp(global_class_logits_t)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _task_posterior_from_shared_features_np(self, shared_features, posterior_router):
        shared_features_t = torch.from_numpy(shared_features).to(self._device, dtype=torch.float32)

        with torch.no_grad():
            if posterior_router == "prototype_activation":
                task_route_prob = self._task_posterior_from_prototype_activation(shared_features_t)
            elif posterior_router == "task_max_proto":
                task_route_prob = self._task_posterior_from_task_max_prototype(shared_features_t)
            else:
                raise ValueError(f"Unsupported feature router: {posterior_router}")

        return task_route_prob.detach().cpu().numpy().astype(np.float32)

    def _class_to_task_np(self, classes):
        classes = np.asarray(classes)
        task_ids = np.full(classes.shape, fill_value=-1, dtype=np.int64)

        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            mask = (classes >= start_idx) & (classes < end_idx)
            task_ids[mask] = task_id

        if np.any(task_ids < 0):
            bad = classes[task_ids < 0][:10]
            raise ValueError(f"Some classes are outside task ranges: {bad}")

        return task_ids

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

    def _collect_eval_pack_np(self, loader):
        self._network.eval()
        all_global_class_logits, all_shared_features, all_targets = [], [], []
        num_tasks = len(self.task_class_ranges)
        expert_logits_chunks = [[] for _ in range(num_tasks)]

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                shared_out = self._network.backbone(inputs, adapter_id=-1, train=False)
                shared_features = shared_out["cls_features"]
                global_class_logits = self._network.fc_shared_cls(shared_features)["logits"][:, : self._total_classes]
                expert_logits_map = self._collect_expert_logits(inputs, list(range(num_tasks))) if num_tasks > 0 else {}

            all_global_class_logits.append(global_class_logits.cpu().numpy().astype(np.float32))
            all_shared_features.append(shared_features.cpu().numpy().astype(np.float32))
            all_targets.append(targets.numpy())
            for task_id in range(num_tasks):
                expert_logits_chunks[task_id].append(expert_logits_map[task_id].cpu().numpy().astype(np.float32))

        global_class_logits_np = np.concatenate(all_global_class_logits, axis=0)
        shared_features_np = np.concatenate(all_shared_features, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        expert_logits_by_task = [
            np.concatenate(task_chunks, axis=0)
            if task_chunks
            else np.zeros((global_class_logits_np.shape[0], 0), dtype=np.float32)
            for task_chunks in expert_logits_chunks
        ]
        return global_class_logits_np, shared_features_np, expert_logits_by_task, y_true

    def _posterior_fusion_probs_np(
        self,
        global_class_logits,
        expert_logits_by_task,
        shared_features=None,
        posterior_router=None,
        return_task_route_prob=False,
    ):
        shared_temperature = max(self.posterior_shared_temperature, 1e-6)
        expert_temperature = max(self.posterior_expert_temperature, 1e-6)
        posterior_router = posterior_router or self.posterior_router

        global_class_prob = self._stable_softmax_np(global_class_logits / shared_temperature).astype(np.float32)

        if not self.task_class_ranges:
            if return_task_route_prob:
                return global_class_prob, global_class_prob, None
            return global_class_prob, global_class_prob

        if posterior_router == "task_logmeanexp":
            task_route_prob = self._task_posterior_from_task_logmeanexp_np(global_class_logits)
        elif posterior_router in {"prototype_activation", "task_max_proto"}:
            if shared_features is None:
                raise ValueError(
                    f"posterior_router={posterior_router} requires shared_features. "
                    "Use _collect_eval_pack_np for prototype-based posterior fusion."
                )
            task_route_prob = self._task_posterior_from_shared_features_np(shared_features, posterior_router)
        else:
            raise ValueError(f"Unknown posterior_router: {posterior_router}")

        routed_expert_prob = np.zeros_like(global_class_prob, dtype=np.float32)
        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            expert_logits = expert_logits_by_task[task_id]
            if expert_logits.shape[1] == 0:
                continue
            local_prob = self._stable_softmax_np(expert_logits / expert_temperature).astype(np.float32)
            routed_expert_prob[:, start_idx:end_idx] = task_route_prob[:, task_id : task_id + 1] * local_prob

        p_final = self.posterior_alpha * global_class_prob + (1.0 - self.posterior_alpha) * routed_expert_prob
        if return_task_route_prob:
            return routed_expert_prob, p_final, task_route_prob
        return routed_expert_prob, p_final

    def eval_task(self):
        global_class_logits_np, shared_features_np, expert_logits_by_task, y_true = self._collect_eval_pack_np(
            self.test_loader
        )

        routed_expert_prob, _ = self._posterior_fusion_probs_np(
            global_class_logits_np,
            expert_logits_by_task,
            shared_features=shared_features_np,
        )

        # NME: Prototype Bank Mixture — always uses task_max_proto routing, no alpha fusion.
        # q_t = softmax(max_{c in task} normalize(z) @ normalize(p_c).T)
        # p_nme(y|x) = Σ_t q_t(x) * 1[y in C_t] * p_expert_t(y|x)
        prototype_mixture_prob, _ = self._posterior_fusion_probs_np(
            global_class_logits_np,
            expert_logits_by_task,
            shared_features=shared_features_np,
            posterior_router="task_max_proto",
        )

        p_moe_pred = self._predict_topk_np(routed_expert_prob)
        prototype_mixture_pred = self._predict_topk_np(prototype_mixture_prob)

        p_moe_accy = self._evaluate(p_moe_pred, y_true)
        prototype_mixture_accy = self._evaluate(prototype_mixture_pred, y_true)
        p_moe_accy = self._drop_topk_metrics(p_moe_accy)
        prototype_mixture_accy = self._drop_topk_metrics(prototype_mixture_accy)
        self._eval_variants = {}

        logging.info(
            (
                "SPiE eval: cnn=p_moe nme=prototype_mixture "
                "posterior_router=%s p_moe_top1=%.2f prototype_mixture_top1=%.2f."
            ),
            self.posterior_router,
            p_moe_accy["top1"],
            prototype_mixture_accy["top1"],
        )

        return p_moe_accy, prototype_mixture_accy
