import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import factory
from utils.data_manager import DataManager


NUM_WORKERS = 8


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(args: Dict[str, Any]) -> None:
    if torch.cuda.is_available():
        args["device"] = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        args["device"] = [torch.device("cpu")]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@torch.no_grad()
def zscore_tensor(values: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    if values.shape[dim] <= 1:
        return torch.zeros_like(values)
    mean = values.mean(dim=dim, keepdim=True)
    std = values.std(dim=dim, keepdim=True, unbiased=False)
    return (values - mean) / std.clamp_min(eps)


@torch.no_grad()
def topk_margin(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] == 0:
        return torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    if values.shape[-1] == 1:
        return values[:, 0]
    top2 = torch.topk(values, k=2, dim=1, largest=True, sorted=True).values
    return top2[:, 0] - top2[:, 1]


@torch.no_grad()
def local_overlap_ratio(shared_slice: torch.Tensor, expert_slice: torch.Tensor, local_topk: int) -> torch.Tensor:
    out = []
    for i in range(shared_slice.shape[0]):
        k = min(local_topk, shared_slice.shape[1], expert_slice.shape[1])
        if k <= 0:
            out.append(torch.zeros((), device=shared_slice.device, dtype=shared_slice.dtype))
            continue
        s_idx = torch.topk(shared_slice[i], k=k, dim=0, largest=True, sorted=False).indices
        e_idx = torch.topk(expert_slice[i], k=k, dim=0, largest=True, sorted=False).indices
        inter = torch.isin(s_idx, e_idx).float().sum()
        out.append(inter / float(k))
    return torch.stack(out, dim=0)


def build_default_strategies() -> List[Dict[str, Any]]:
    return [
        {"name": "shared_only", "kind": "shared_only"},
        {"name": "v15_js_rerank", "kind": "candidate_rerank", "candidate_width": 5, "bonus": "js", "bonus_weight": 0.50},
        {"name": "cand_energy_rerank", "kind": "candidate_rerank", "candidate_width": 8, "bonus": "energy", "bonus_weight": 0.50},
        {"name": "cand_hybrid_rerank", "kind": "candidate_rerank", "candidate_width": 8, "bonus": "hybrid", "bonus_weight": 0.50},
        {"name": "v16_single", "kind": "single_task", "task_score": "shared_energy", "shared_w": 0.50, "expert_w": 0.50},
        {"name": "v16_veto_gap03_e0", "kind": "single_task_veto", "task_score": "shared_energy", "shared_w": 0.50, "expert_w": 0.50, "score_gap": 0.30, "energy_floor": 0.00},
        {"name": "v16_veto_gap06_e0", "kind": "single_task_veto", "task_score": "shared_energy", "shared_w": 0.50, "expert_w": 0.50, "score_gap": 0.60, "energy_floor": 0.00},
        {"name": "expert_only_selected", "kind": "single_task", "task_score": "shared_energy", "shared_w": 0.00, "expert_w": 1.00},
        {"name": "top2_union_linear", "kind": "topm_union_linear", "task_score": "hybrid", "topm": 2, "shared_w": 1.00, "expert_w": 0.35, "task_bonus": 0.25},
        {"name": "top3_union_linear", "kind": "topm_union_linear", "task_score": "hybrid", "topm": 3, "shared_w": 1.00, "expert_w": 0.35, "task_bonus": 0.20},
        {"name": "top2_union_residual", "kind": "topm_union_residual", "task_score": "hybrid", "topm": 2, "residual_w": 0.40, "task_bonus": 0.15},
        {"name": "top2_union_rrf", "kind": "topm_union_rrf", "task_score": "hybrid", "topm": 2, "beta": 6.0, "rrf_k": 10.0},
        {"name": "soft_tasks_linear", "kind": "soft_tasks_linear", "task_score": "hybrid", "topm": 3, "beta": 5.0, "shared_w": 1.00, "expert_w": 0.30},
        {"name": "soft_tasks_overlap_bonus", "kind": "soft_tasks_overlap", "task_score": "hybrid", "topm": 3, "beta": 5.0, "bonus_weight": 0.75},
        {"name": "top2_union_overlap_guard", "kind": "topm_union_overlap_guard", "task_score": "hybrid", "topm": 2, "shared_w": 1.00, "expert_w": 0.30, "min_overlap": 0.34},
    ]


class SweepEvaluator:
    def __init__(self, model, topk: int):
        self.model = model
        self.topk = topk
        self.max_candidate_width = max(8, topk, int(getattr(model, "energy_topk", topk)))

    @torch.no_grad()
    def _predict_from_scores(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.topk, scores.shape[1])
        pred = torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices
        if k < self.topk:
            pad = torch.full((pred.shape[0], self.topk - k), -1, device=pred.device, dtype=pred.dtype)
            pred = torch.cat([pred, pad], dim=1)
        return pred

    @torch.no_grad()
    def _task_score(self, cache: Dict[str, Any], row_idx: int, task_id: int, kind: str) -> torch.Tensor:
        if kind == "shared":
            return cache["shared_task_max_map"][task_id][row_idx]
        if kind == "energy":
            return cache["energy_score_map"][task_id][row_idx]
        if kind == "js":
            return cache["task_similarity_map"][task_id][row_idx]
        if kind == "margin":
            return cache["shared_task_margin_map"][task_id][row_idx]
        if kind == "shared_energy":
            a = cache["shared_task_max_z_map"][task_id][row_idx]
            b = cache["energy_score_map"][task_id][row_idx]
            return 0.5 * a + 0.5 * b
        if kind == "hybrid":
            a = cache["shared_task_max_z_map"][task_id][row_idx]
            b = cache["energy_score_map"][task_id][row_idx]
            c = cache["task_similarity_map"][task_id][row_idx]
            return 0.40 * a + 0.35 * b + 0.25 * c
        raise ValueError(f"Unknown task score kind: {kind}")

    @torch.no_grad()
    def _build_batch_cache(self, inputs: torch.Tensor) -> Dict[str, Any]:
        model = self.model
        shared_logits = model._shared_cls_logits(inputs)
        candidate_width = min(self.max_candidate_width, shared_logits.shape[1])
        topcand = torch.topk(shared_logits, k=candidate_width, dim=1, largest=True, sorted=True).indices
        gate_width = min(model.energy_topk, topcand.shape[1])
        gate_topk = topcand[:, :gate_width]
        candidate_tasks, unique_task_ids = model._select_candidate_tasks(gate_topk)
        expert_logits_map = model._collect_expert_logits(inputs, unique_task_ids)

        shared_task_max_map = {}
        shared_task_max_z_map = {}
        shared_task_margin_map = {}
        energy_score_map = {}
        task_similarity_map = {}
        overlap_ratio_map = {}

        all_shared_task_max = []
        all_task_order = []

        for task_id in unique_task_ids:
            start_idx, end_idx = model.task_class_ranges[task_id]
            shared_slice = shared_logits[:, start_idx:end_idx]
            expert_slice = expert_logits_map[task_id]

            shared_task_max_map[task_id] = torch.max(shared_slice, dim=1).values
            shared_task_margin_map[task_id] = topk_margin(shared_slice)
            raw_energy = model._energy_from_logits(expert_slice)
            mean_in, std_in = model._network.get_expert_energy_stats(task_id)
            energy_score_map[task_id] = (
                raw_energy - mean_in.to(device=raw_energy.device, dtype=raw_energy.dtype)
            ) / (std_in.to(device=raw_energy.device, dtype=raw_energy.dtype) + 1e-6)
            task_similarity_map[task_id] = model._batch_task_local_similarity(shared_slice, expert_slice)
            overlap_ratio_map[task_id] = local_overlap_ratio(shared_slice, expert_slice, model.verifier_local_topk)
            all_shared_task_max.append(shared_task_max_map[task_id])
            all_task_order.append(task_id)

        if all_shared_task_max:
            stacked_max = torch.stack(all_shared_task_max, dim=1)
            stacked_max_z = zscore_tensor(stacked_max, dim=1)
            for col, task_id in enumerate(all_task_order):
                shared_task_max_z_map[task_id] = stacked_max_z[:, col]

        return {
            "shared_logits": shared_logits,
            "shared_topk": model._predict_topk(shared_logits),
            "topcand": topcand,
            "candidate_tasks": candidate_tasks,
            "unique_task_ids": unique_task_ids,
            "expert_logits_map": expert_logits_map,
            "shared_task_max_map": shared_task_max_map,
            "shared_task_max_z_map": shared_task_max_z_map,
            "shared_task_margin_map": shared_task_margin_map,
            "energy_score_map": energy_score_map,
            "task_similarity_map": task_similarity_map,
            "overlap_ratio_map": overlap_ratio_map,
        }

    @torch.no_grad()
    def _candidate_rerank(self, cache: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
        shared_logits = cache["shared_logits"]
        topcand = cache["topcand"]
        predicts = cache["shared_topk"].clone()
        candidate_width = min(int(cfg.get("candidate_width", self.topk)), topcand.shape[1])
        bonus_name = cfg.get("bonus", "js")
        bonus_weight = float(cfg.get("bonus_weight", 0.5))

        for i in range(shared_logits.shape[0]):
            cand = topcand[i, :candidate_width]
            scores = []
            for cls_idx in cand.tolist():
                cls_idx = int(cls_idx)
                task_id = self.model._class_to_task_id(cls_idx)
                bonus = 0.0
                if bonus_name == "js":
                    bonus = cache["task_similarity_map"][task_id][i]
                elif bonus_name == "energy":
                    bonus = cache["energy_score_map"][task_id][i]
                elif bonus_name == "hybrid":
                    bonus = self._task_score(cache, i, task_id, "hybrid")
                else:
                    raise ValueError(f"Unknown bonus: {bonus_name}")
                scores.append(shared_logits[i, cls_idx] + bonus_weight * bonus)
            scores = torch.stack(scores, dim=0)
            order = torch.argsort(scores, descending=True)
            ranked = cand[order]
            fill = min(self.topk, ranked.numel())
            predicts[i, :fill] = ranked[:fill]
        return predicts

    @torch.no_grad()
    def _single_task(self, cache: Dict[str, Any], cfg: Dict[str, Any], allow_veto: bool) -> torch.Tensor:
        shared_logits = cache["shared_logits"]
        predicts = torch.full((shared_logits.shape[0], self.topk), -1, device=shared_logits.device, dtype=torch.long)
        shared_fallback = cache["shared_topk"]
        score_kind = cfg.get("task_score", "shared_energy")
        shared_w = float(cfg.get("shared_w", 0.5))
        expert_w = float(cfg.get("expert_w", 0.5))
        score_gap_thr = float(cfg.get("score_gap", -1e9))
        energy_floor = float(cfg.get("energy_floor", -1e9))

        for i, row_tasks in enumerate(cache["candidate_tasks"]):
            if not row_tasks:
                predicts[i] = shared_fallback[i]
                continue
            task_scores = torch.stack([self._task_score(cache, i, t, score_kind) for t in row_tasks], dim=0)
            order = torch.argsort(task_scores, descending=True)
            best_pos = int(order[0].item())
            best_task = row_tasks[best_pos]

            if allow_veto:
                gap = torch.tensor(float("inf"), device=task_scores.device, dtype=task_scores.dtype)
                if task_scores.numel() >= 2:
                    gap = task_scores[order[0]] - task_scores[order[1]]
                if gap.item() < score_gap_thr or cache["energy_score_map"][best_task][i].item() < energy_floor:
                    predicts[i] = shared_fallback[i]
                    continue

            start_idx, end_idx = self.model.task_class_ranges[best_task]
            shared_slice = zscore_tensor(cache["shared_logits"][i, start_idx:end_idx].unsqueeze(0), dim=1)[0]
            expert_slice = zscore_tensor(cache["expert_logits_map"][best_task][i].unsqueeze(0), dim=1)[0]
            final_scores = shared_w * shared_slice + expert_w * expert_slice
            order_local = torch.argsort(final_scores, descending=True)
            fill = min(self.topk, order_local.numel())
            predicts[i, :fill] = order_local[:fill] + start_idx
        return predicts

    @torch.no_grad()
    def _topm_union_linear(
        self,
        cache: Dict[str, Any],
        cfg: Dict[str, Any],
        residual: bool = False,
        soft: bool = False,
        overlap_guard: bool = False,
    ) -> torch.Tensor:
        shared_logits = cache["shared_logits"]
        predicts = torch.full((shared_logits.shape[0], self.topk), -1, device=shared_logits.device, dtype=torch.long)
        score_kind = cfg.get("task_score", "hybrid")
        topm = int(cfg.get("topm", 2))
        shared_w = float(cfg.get("shared_w", 1.0))
        expert_w = float(cfg.get("expert_w", 0.35))
        residual_w = float(cfg.get("residual_w", 0.40))
        task_bonus = float(cfg.get("task_bonus", 0.20))
        beta = float(cfg.get("beta", 5.0))
        min_overlap = float(cfg.get("min_overlap", 0.0))

        for i, row_tasks in enumerate(cache["candidate_tasks"]):
            if not row_tasks:
                predicts[i] = cache["shared_topk"][i]
                continue
            row_scores = torch.stack([self._task_score(cache, i, t, score_kind) for t in row_tasks], dim=0)
            order = torch.argsort(row_scores, descending=True)
            chosen_pos = order[: min(topm, order.numel())]
            chosen_tasks = [row_tasks[int(p.item())] for p in chosen_pos]

            final = shared_logits[i].clone()
            if soft:
                weights = torch.softmax(beta * row_scores[chosen_pos], dim=0)
            else:
                weights = torch.ones(len(chosen_tasks), device=final.device, dtype=final.dtype)

            for w, task_id in zip(weights.tolist(), chosen_tasks):
                start_idx, end_idx = self.model.task_class_ranges[task_id]
                shared_slice = zscore_tensor(shared_logits[i, start_idx:end_idx].unsqueeze(0), dim=1)[0]
                expert_slice = zscore_tensor(cache["expert_logits_map"][task_id][i].unsqueeze(0), dim=1)[0]
                overlap = cache["overlap_ratio_map"][task_id][i].item()
                if overlap_guard and overlap < min_overlap:
                    continue
                if residual:
                    local_scores = shared_slice + residual_w * (expert_slice - expert_slice.mean())
                else:
                    local_scores = shared_w * shared_slice + expert_w * expert_slice
                bonus = task_bonus * self._task_score(cache, i, task_id, score_kind)
                final[start_idx:end_idx] = final[start_idx:end_idx] + float(w) * (local_scores + bonus)

            predicts[i] = self._predict_from_scores(final.unsqueeze(0))[0]
        return predicts

    @torch.no_grad()
    def _topm_union_rrf(self, cache: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
        shared_logits = cache["shared_logits"]
        predicts = torch.full((shared_logits.shape[0], self.topk), -1, device=shared_logits.device, dtype=torch.long)
        score_kind = cfg.get("task_score", "hybrid")
        topm = int(cfg.get("topm", 2))
        beta = float(cfg.get("beta", 6.0))
        rrf_k = float(cfg.get("rrf_k", 10.0))

        for i, row_tasks in enumerate(cache["candidate_tasks"]):
            if not row_tasks:
                predicts[i] = cache["shared_topk"][i]
                continue

            row_scores = torch.stack([self._task_score(cache, i, t, score_kind) for t in row_tasks], dim=0)
            order = torch.argsort(row_scores, descending=True)
            chosen_pos = order[: min(topm, order.numel())]
            chosen_tasks = [row_tasks[int(p.item())] for p in chosen_pos]
            task_weights = torch.softmax(beta * row_scores[chosen_pos], dim=0)

            final = torch.zeros_like(shared_logits[i])
            shared_rank = torch.argsort(torch.argsort(-shared_logits[i]))
            final += 1.0 / (rrf_k + shared_rank.float())

            for wt, task_id in zip(task_weights.tolist(), chosen_tasks):
                start_idx, end_idx = self.model.task_class_ranges[task_id]
                expert_slice = cache["expert_logits_map"][task_id][i]
                local_rank = torch.argsort(torch.argsort(-expert_slice))
                final[start_idx:end_idx] += float(wt) * (1.0 / (rrf_k + local_rank.float()))

            predicts[i] = self._predict_from_scores(final.unsqueeze(0))[0]
        return predicts

    @torch.no_grad()
    def _soft_tasks_overlap(self, cache: Dict[str, Any], cfg: Dict[str, Any]) -> torch.Tensor:
        shared_logits = cache["shared_logits"]
        predicts = torch.full((shared_logits.shape[0], self.topk), -1, device=shared_logits.device, dtype=torch.long)
        score_kind = cfg.get("task_score", "hybrid")
        topm = int(cfg.get("topm", 3))
        beta = float(cfg.get("beta", 5.0))
        bonus_weight = float(cfg.get("bonus_weight", 0.75))

        for i, row_tasks in enumerate(cache["candidate_tasks"]):
            if not row_tasks:
                predicts[i] = cache["shared_topk"][i]
                continue
            row_scores = torch.stack([self._task_score(cache, i, t, score_kind) for t in row_tasks], dim=0)
            order = torch.argsort(row_scores, descending=True)
            chosen_pos = order[: min(topm, order.numel())]
            chosen_tasks = [row_tasks[int(p.item())] for p in chosen_pos]
            weights = torch.softmax(beta * row_scores[chosen_pos], dim=0)
            final = shared_logits[i].clone()

            for wt, task_id in zip(weights.tolist(), chosen_tasks):
                start_idx, end_idx = self.model.task_class_ranges[task_id]
                k = min(self.model.verifier_local_topk, end_idx - start_idx)
                if k <= 0:
                    continue
                s_idx = torch.topk(shared_logits[i, start_idx:end_idx], k=k, dim=0, largest=True, sorted=False).indices
                e_idx = torch.topk(cache["expert_logits_map"][task_id][i], k=k, dim=0, largest=True, sorted=False).indices
                support = torch.unique(torch.cat([s_idx, e_idx], dim=0), sorted=False)
                final[start_idx + support] += float(wt) * bonus_weight * self._task_score(cache, i, task_id, score_kind)

            predicts[i] = self._predict_from_scores(final.unsqueeze(0))[0]
        return predicts

    @torch.no_grad()
    def predict_batch(self, inputs: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
        cache = self._build_batch_cache(inputs)
        kind = cfg["kind"]
        if kind == "shared_only":
            return cache["shared_topk"]
        if kind == "candidate_rerank":
            return self._candidate_rerank(cache, cfg)
        if kind == "single_task":
            return self._single_task(cache, cfg, allow_veto=False)
        if kind == "single_task_veto":
            return self._single_task(cache, cfg, allow_veto=True)
        if kind == "topm_union_linear":
            return self._topm_union_linear(cache, cfg, residual=False, soft=False, overlap_guard=False)
        if kind == "topm_union_residual":
            return self._topm_union_linear(cache, cfg, residual=True, soft=False, overlap_guard=False)
        if kind == "topm_union_rrf":
            return self._topm_union_rrf(cache, cfg)
        if kind == "soft_tasks_linear":
            return self._topm_union_linear(cache, cfg, residual=False, soft=True, overlap_guard=False)
        if kind == "soft_tasks_overlap":
            return self._soft_tasks_overlap(cache, cfg)
        if kind == "topm_union_overlap_guard":
            return self._topm_union_linear(cache, cfg, residual=False, soft=False, overlap_guard=True)
        raise ValueError(f"Unknown strategy kind: {kind}")


@torch.no_grad()
def build_model_for_checkpoint(args: Dict[str, Any], checkpoint: Dict[str, Any], data_manager: DataManager):
    model = factory.get_model(args["model_name"], args)
    target_task = int(checkpoint["tasks"])

    for task_id in range(target_task + 1):
        model._cur_task += 1
        model._total_classes = model._known_classes + data_manager.get_task_size(task_id)
        current_task_size = model._total_classes - model._known_classes
        model.task_class_ranges.append((model._known_classes, model._total_classes))
        model._network.update_fc(current_task_size)
        model._network.append_expert_head(current_task_size)
        if model._should_reset_task_modules():
            backbone = model._backbone_module()
            backbone.reset_task_modules()
            if hasattr(backbone, "adapter_update"):
                backbone.adapter_update()
        model._known_classes = model._total_classes

    model._network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model._network.to(model._device)
    model._network.eval()
    model.data_manager = data_manager

    test_dataset = data_manager.get_dataset(np.arange(0, model._total_classes), source="test", mode="test")
    model.test_loader = DataLoader(
        test_dataset,
        batch_size=model.batch_size,
        shuffle=False,
        num_workers=int(args.get("eval_num_workers", NUM_WORKERS)),
    )
    return model


def evaluate_strategies(model, strategies: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    evaluator = SweepEvaluator(model, topk=model.topk)
    outputs = {cfg["name"]: [] for cfg in strategies}
    labels = []

    for _, inputs, targets in model.test_loader:
        inputs = inputs.to(model._device)
        labels.append(targets.cpu().numpy())
        for cfg in strategies:
            pred = evaluator.predict_batch(inputs, cfg)
            outputs[cfg["name"]].append(pred.cpu().numpy())

    y_true = np.concatenate(labels, axis=0)
    results = {}
    for cfg in strategies:
        name = cfg["name"]
        y_pred = np.concatenate(outputs[name], axis=0)
        results[name] = model._evaluate(y_pred, y_true)
    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    rows = []
    for name, metrics in results.items():
        rows.append((name, metrics["top1"], metrics.get("top5", None)))
    rows.sort(key=lambda x: (-x[1], x[0]))

    print("\n===== strategy ranking =====")
    for name, top1, top5 in rows:
        top5_str = "  top5=  n/a" if top5 is None else f"  top5={top5:6.2f}"
        print(f"{name:28s} top1={top1:6.2f}{top5_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-only sweep for SPIE v16 inference strategies.")
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch size override.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers for evaluation.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cli = parse_args()

    with open(cli.config, "r") as f:
        args = json.load(f)

    args = copy.deepcopy(args)
    if cli.batch_size is not None:
        args["batch_size"] = int(cli.batch_size)
    args["eval_num_workers"] = int(cli.num_workers)

    set_device(args)
    seed = cli.seed if cli.seed is not None else (
        args["seed"][0] if isinstance(args.get("seed", 1), list) else int(args.get("seed", 1))
    )
    set_seed(int(seed))

    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        int(seed),
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    model = build_model_for_checkpoint(args, checkpoint, data_manager)
    strategies = build_default_strategies()
    results = evaluate_strategies(model, strategies)
    print_results(results)

    if cli.output:
        out_path = Path(cli.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(to_builtin(results), f, indent=2)
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
