#!/usr/bin/env python3
"""
Post-hoc prior / fusion sweep for SPIE v14.

What it does
------------
1. Loads a finished SPIE v14 checkpoint.
2. Rebuilds the shared classifier + expert detector stacks to match the checkpoint.
3. Runs the test set once and caches:
   - shared global logits
   - every expert detector's task score
   - every expert detector's detailed stats
4. Sweeps a broad family of post-hoc shared/task-prior fusion rules without retraining.
5. Saves CSV summaries sorted by accuracy.

This script intentionally treats the test set as an oracle tuning playground.
Use it for analysis / upper-bound exploration, not as a clean final protocol.

Run from the repository root, e.g.
python spie_v14_prior_sweep.py \
  --config exps/spie_v14_inr.json \
  --checkpoint logs/xxx/checkpoints/task_9.pkl \
  --output-dir sweep_spie_v14
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import factory
from utils.data_manager import DataManager


EPS = 1e-12
STAT_NAMES = [
    "center_max",
    "center_mean",
    "rep_max",
    "rep_mean",
    "neg_maha",
    "ood_penalty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPIE v14 post-hoc prior sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config used for training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to task_*.pkl checkpoint.")
    parser.add_argument("--output-dir", type=str, default="prior_sweep_out", help="Directory to save cache + CSVs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for extraction.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers for test extraction.")
    parser.add_argument("--force-recache", action="store_true", help="Recompute cached scores even if cache exists.")
    parser.add_argument(
        "--cache-name",
        type=str,
        default="cached_prior_scores.npz",
        help="Filename for the cached shared/task score tensors under output-dir.",
    )
    parser.add_argument(
        "--grid-preset",
        type=str,
        default="medium",
        choices=["quick", "medium", "full"],
        help="Hyperparameter sweep size.",
    )
    parser.add_argument(
        "--candidate-task-from-topk",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for candidate task source widths, e.g. --candidate-task-from-topk 2 3 5",
    )
    parser.add_argument(
        "--class-rerank-topk",
        type=int,
        nargs="*",
        default=None,
        help="Optional override for class rerank widths, e.g. --class-rerank-topk 3 5",
    )
    return parser.parse_args()


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True).clip(min=EPS)


def logsumexp_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    stable = x - x_max
    out = x_max + np.log(np.sum(np.exp(stable), axis=axis, keepdims=True) + EPS)
    return np.squeeze(out, axis=axis)


def rank_desc(x: np.ndarray, topk: int) -> np.ndarray:
    topk = min(topk, x.shape[1])
    order = np.argsort(-x, axis=1)
    return order[:, :topk]


def top1_top2_gap(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.squeeze(x[:, :1], axis=1)
    part = np.partition(x, kth=x.shape[1] - 2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    return top1 - top2


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.clip(std, a_min=EPS, a_max=None)


def minmax_rows(x: np.ndarray) -> np.ndarray:
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    return (x - x_min) / np.clip(x_max - x_min, a_min=EPS, a_max=None)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_scalar_config_value(args: Dict[str, Any], key: str) -> None:
    value = args.get(key)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"Expected config field '{key}' to be a scalar or a single-item list/tuple, got {value!r}."
            )
        args[key] = value[0]


def normalize_args(cfg: Dict[str, Any], device_str: str, batch_size_override: int | None) -> Dict[str, Any]:
    args = dict(cfg)
    if "memory_size" not in args:
        args["memory_size"] = 0
    if "fixed_memory" not in args:
        args["fixed_memory"] = False
    if "memory_per_class" not in args:
        args["memory_per_class"] = None

    _normalize_scalar_config_value(args, "seed")
    _normalize_scalar_config_value(args, "init_cls")
    _normalize_scalar_config_value(args, "increment")

    normalized_device = str(device_str).strip()
    if normalized_device.isdigit():
        normalized_device = f"cuda:{normalized_device}"
    requested = torch.device(normalized_device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable. Falling back to CPU.")
        requested = torch.device("cpu")
    args["device"] = [requested]

    if batch_size_override is not None:
        args["batch_size"] = int(batch_size_override)

    return args


def rebuild_spie_v14_to_checkpoint(learner, data_manager: DataManager, checkpoint: Dict[str, Any]) -> None:
    if "tasks" not in checkpoint:
        raise KeyError("Checkpoint is missing 'tasks'. Expected a BaseLearner task_*.pkl checkpoint.")

    last_task = int(checkpoint["tasks"])
    seen_classes = 0
    learner.task_class_ranges = []
    backbone = learner._backbone_module()

    for task_id in range(last_task + 1):
        task_size = int(data_manager.get_task_size(task_id))
        learner.task_class_ranges.append((seen_classes, seen_classes + task_size))
        learner._network.update_fc(task_size)
        seen_classes += task_size

        backbone.adapter_update()
        if task_id < last_task:
            backbone.reset_task_modules()

    learner._cur_task = last_task
    learner._total_classes = int(checkpoint.get("total_classes", seen_classes))
    learner._known_classes = learner._total_classes

    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = learner._network.load_state_dict(state_dict, strict=False)
    if missing:
        print("[warn] Missing state_dict keys:")
        for key in missing:
            print("   ", key)
    if unexpected:
        print("[warn] Unexpected state_dict keys:")
        for key in unexpected:
            print("   ", key)


def build_test_loader(data_manager: DataManager, total_classes: int, batch_size: int, num_workers: int) -> DataLoader:
    test_dataset = data_manager.get_dataset(np.arange(total_classes), source="test", mode="test")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def cache_paths(output_dir: str, cache_name: str) -> Tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, out_dir / cache_name


def build_cache_metadata(
    config_path: str,
    checkpoint_path: str,
    norm_args: Dict[str, Any],
    learner,
) -> Dict[str, Any]:
    checkpoint_stat = os.stat(checkpoint_path)
    return {
        "cache_version": 1,
        "config_path": str(Path(config_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "dataset": str(norm_args["dataset"]),
        "seed": int(norm_args["seed"]),
        "init_cls": int(norm_args["init_cls"]),
        "increment": int(norm_args["increment"]),
        "total_classes": int(learner._total_classes),
        "num_tasks": int(len(learner.task_class_ranges)),
    }


def _read_cache_metadata(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "cache_metadata" not in data.files:
        return {}
    raw = data["cache_metadata"]
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if not isinstance(raw, dict):
        return {}
    return raw


def extract_or_load_scores(
    learner,
    loader: DataLoader,
    force_recache: bool,
    cache_file: Path,
    cache_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    if cache_file.exists() and not force_recache:
        data = np.load(cache_file, allow_pickle=True)
        stored_metadata = _read_cache_metadata(data)
        if stored_metadata != cache_metadata:
            print(f"[warn] Ignoring stale cache: {cache_file}")
            print(f"[warn] Cached metadata: {stored_metadata if stored_metadata else '<missing>'}")
            print(f"[warn] Expected metadata: {cache_metadata}")
        else:
            print(f"[info] Loading cache: {cache_file}")
            return {
                "shared_logits": data["shared_logits"],
                "targets": data["targets"],
                "task_starts": data["task_starts"],
                "task_ends": data["task_ends"],
                "task_scores": data["task_scores"],
                "task_stats": data["task_stats"],
                "num_tasks": int(data["num_tasks"]),
                "total_classes": int(data["total_classes"]),
            }

    print("[info] Extracting shared logits + task prior scores from the test set...")
    learner._network.eval()
    device = learner._device
    task_ids = list(range(len(learner.task_class_ranges)))
    shared_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    score_chunks: List[np.ndarray] = []
    stats_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            shared_logits = learner._shared_cls_logits(inputs)
            ood_out = learner._network.forward_multi_expert_ood_scores(inputs, task_ids)
            task_scores = ood_out["scores"].transpose(0, 1).contiguous()
            task_stats = ood_out["stats"].permute(1, 0, 2).contiguous()

            shared_chunks.append(shared_logits.detach().cpu().numpy().astype(np.float32))
            target_chunks.append(targets.numpy().astype(np.int64))
            score_chunks.append(task_scores.detach().cpu().numpy().astype(np.float32))
            stats_chunks.append(task_stats.detach().cpu().numpy().astype(np.float32))

    shared_logits_np = np.concatenate(shared_chunks, axis=0)
    targets_np = np.concatenate(target_chunks, axis=0)
    task_scores_np = np.concatenate(score_chunks, axis=0)
    task_stats_np = np.concatenate(stats_chunks, axis=0)
    task_starts = np.array([start for start, _ in learner.task_class_ranges], dtype=np.int64)
    task_ends = np.array([end for _, end in learner.task_class_ranges], dtype=np.int64)

    payload = {
        "shared_logits": shared_logits_np,
        "targets": targets_np,
        "task_starts": task_starts,
        "task_ends": task_ends,
        "task_scores": task_scores_np,
        "task_stats": task_stats_np,
        "num_tasks": np.int64(len(task_ids)),
        "total_classes": np.int64(learner._total_classes),
        "cache_metadata": np.array(cache_metadata, dtype=object),
    }
    np.savez_compressed(cache_file, **payload)
    print(f"[info] Saved cache: {cache_file}")

    return {
        "shared_logits": shared_logits_np,
        "targets": targets_np,
        "task_starts": task_starts,
        "task_ends": task_ends,
        "task_scores": task_scores_np,
        "task_stats": task_stats_np,
        "num_tasks": len(task_ids),
        "total_classes": learner._total_classes,
    }


def build_class_to_task(task_starts: np.ndarray, task_ends: np.ndarray) -> np.ndarray:
    total_classes = int(task_ends[-1])
    arr = np.full((total_classes,), -1, dtype=np.int64)
    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        arr[start:end] = task_id
    return arr


def candidate_tasks_from_topk_classes(
    shared_topk: np.ndarray,
    class_to_task: np.ndarray,
    topk_width: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    width = min(topk_width, shared_topk.shape[1])
    for row in shared_topk[:, :width]:
        seen = set()
        tasks = []
        for class_idx in row.tolist():
            task_id = int(class_to_task[class_idx])
            if task_id not in seen:
                seen.add(task_id)
                tasks.append(task_id)
        out.append(np.array(tasks, dtype=np.int64))
    return out


def top1_accuracy(pred: np.ndarray, targets: np.ndarray) -> float:
    return float((pred == targets).mean() * 100.0)


def grouped_correction_stats(pred: np.ndarray, baseline_pred: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    baseline_correct = baseline_pred == targets
    pred_correct = pred == targets
    rescued = float(np.logical_and(~baseline_correct, pred_correct).sum())
    hurt = float(np.logical_and(baseline_correct, ~pred_correct).sum())
    total = float(targets.shape[0])
    return {
        "rescued_count": rescued,
        "hurt_count": hurt,
        "rescued_rate": rescued / total,
        "hurt_rate": hurt / total,
    }


def choose_task_by_score(candidate_tasks: np.ndarray, scores: np.ndarray, fallback_task: int) -> int:
    if candidate_tasks.size == 0:
        return int(fallback_task)
    local_idx = int(np.argmax(scores[candidate_tasks]))
    return int(candidate_tasks[local_idx])


def build_metric_bank(task_scores: np.ndarray, task_stats: np.ndarray) -> Dict[str, np.ndarray]:
    stats = {name: task_stats[:, :, idx] for idx, name in enumerate(STAT_NAMES)}

    bank = {
        "task_score_raw": task_scores,
        "task_score_z": zscore_rows(task_scores),
        "task_score_minmax": minmax_rows(task_scores),
        "center_max": stats["center_max"],
        "center_max_z": zscore_rows(stats["center_max"]),
        "center_mean": stats["center_mean"],
        "center_mean_z": zscore_rows(stats["center_mean"]),
        "rep_max": stats["rep_max"],
        "rep_max_z": zscore_rows(stats["rep_max"]),
        "rep_mean": stats["rep_mean"],
        "rep_mean_z": zscore_rows(stats["rep_mean"]),
        "neg_maha": stats["neg_maha"],
        "neg_maha_z": zscore_rows(stats["neg_maha"]),
        "neg_ood": -stats["ood_penalty"],
        "neg_ood_z": zscore_rows(-stats["ood_penalty"]),
        "repmax_minus_ood": stats["rep_max"] - stats["ood_penalty"],
        "repmax_minus_ood_z": zscore_rows(stats["rep_max"] - stats["ood_penalty"]),
        "repmean_minus_ood": stats["rep_mean"] - stats["ood_penalty"],
        "repmean_minus_ood_z": zscore_rows(stats["rep_mean"] - stats["ood_penalty"]),
        "repmax_plus_negmaha": stats["rep_max"] + stats["neg_maha"],
        "repmax_plus_negmaha_z": zscore_rows(stats["rep_max"] + stats["neg_maha"]),
        "centermax_minus_ood": stats["center_max"] - stats["ood_penalty"],
        "centermax_minus_ood_z": zscore_rows(stats["center_max"] - stats["ood_penalty"]),
    }
    return {key: value.astype(np.float32) for key, value in bank.items()}


def precompute_metrics(
    shared_logits: np.ndarray,
    task_scores: np.ndarray,
    task_stats: np.ndarray,
    task_starts: np.ndarray,
    task_ends: np.ndarray,
) -> Dict[str, Any]:
    num_samples = shared_logits.shape[0]
    num_tasks = task_scores.shape[1]
    block_lse = np.zeros((num_samples, num_tasks), dtype=np.float32)
    block_max = np.zeros((num_samples, num_tasks), dtype=np.float32)
    block_margin = np.zeros((num_samples, num_tasks), dtype=np.float32)
    shared_local_top1 = np.zeros((num_samples, num_tasks), dtype=np.int64)

    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        local_slice = shared_logits[:, start:end]
        block_lse[:, task_id] = logsumexp_np(local_slice, axis=1).astype(np.float32)
        block_max[:, task_id] = np.max(local_slice, axis=1).astype(np.float32)
        block_margin[:, task_id] = top1_top2_gap(local_slice).astype(np.float32)
        shared_local_top1[:, task_id] = local_slice.argmax(axis=1) + start

    return {
        "block_lse": block_lse,
        "block_max": block_max,
        "block_margin": block_margin,
        "shared_local_top1": shared_local_top1,
        "metric_bank": build_metric_bank(task_scores, task_stats),
    }


def method_baseline_shared(shared_logits: np.ndarray) -> np.ndarray:
    return shared_logits.argmax(axis=1).astype(np.int64)


def method_oracle_shared_local(targets: np.ndarray, class_to_task: np.ndarray, shared_local_top1: np.ndarray) -> np.ndarray:
    gt_task = class_to_task[targets]
    return shared_local_top1[np.arange(targets.shape[0]), gt_task]


def method_softmax_scale_then_shared(
    shared_logits: np.ndarray,
    metric: np.ndarray,
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    alpha: float,
    temperature: float,
    mode: str,
) -> np.ndarray:
    weights = softmax_np(metric / max(float(temperature), 1e-6), axis=1)
    uniform = 1.0 / max(metric.shape[1], 1)
    adjusted = shared_logits.copy()

    if mode == "mul_centered":
        task_values = np.clip(1.0 + float(alpha) * (weights - uniform), a_min=0.5, a_max=1.5)
    elif mode == "mul_softmax":
        task_values = np.clip(1.0 + float(alpha) * weights, a_min=0.5, a_max=2.0)
    elif mode == "add_softmax":
        task_values = float(alpha) * weights
    else:
        raise ValueError(f"Unknown scale mode: {mode}")

    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        if mode.startswith("mul_"):
            adjusted[:, start:end] *= task_values[:, task_id][:, None]
        else:
            adjusted[:, start:end] += task_values[:, task_id][:, None]
    return adjusted.argmax(axis=1).astype(np.int64)


def method_task_bonus_all_classes(
    shared_logits: np.ndarray,
    metric: np.ndarray,
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    alpha: float,
) -> np.ndarray:
    adjusted = shared_logits.copy()
    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        adjusted[:, start:end] += float(alpha) * metric[:, task_id][:, None]
    return adjusted.argmax(axis=1).astype(np.int64)


def method_class_bonus_candidate_rerank(
    shared_logits: np.ndarray,
    shared_topk: np.ndarray,
    metric: np.ndarray,
    class_to_task: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    alpha: float,
    rerank_topk: int,
) -> np.ndarray:
    pred = shared_logits.argmax(axis=1).astype(np.int64)
    width = min(rerank_topk, shared_topk.shape[1])
    for i in range(shared_logits.shape[0]):
        candidate_classes = shared_topk[i, :width]
        scores = shared_logits[i, candidate_classes].copy()
        candidate_task_set = set(candidate_tasks_by_k[i].tolist())
        for j, class_idx in enumerate(candidate_classes.tolist()):
            task_id = int(class_to_task[class_idx])
            if task_id in candidate_task_set:
                scores[j] += float(alpha) * metric[i, task_id]
        pred[i] = int(candidate_classes[int(np.argmax(scores))])
    return pred


def method_select_task_then_shared(
    block_scores: np.ndarray,
    metric: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    shared_logits: np.ndarray,
    alpha: float,
) -> np.ndarray:
    pred = np.zeros((shared_logits.shape[0],), dtype=np.int64)
    baseline_task = block_scores.argmax(axis=1)
    for i in range(shared_logits.shape[0]):
        candidate_tasks = candidate_tasks_by_k[i]
        if candidate_tasks.size == 0:
            chosen_task = int(baseline_task[i])
        else:
            score_vec = block_scores[i].copy()
            score_vec[candidate_tasks] += float(alpha) * metric[i, candidate_tasks]
            chosen_task = choose_task_by_score(candidate_tasks, score_vec, int(baseline_task[i]))
        start, end = int(task_starts[chosen_task]), int(task_ends[chosen_task])
        pred[i] = start + int(np.argmax(shared_logits[i, start:end]))
    return pred


def method_gated_select_task_then_shared(
    block_scores: np.ndarray,
    metric: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    shared_logits: np.ndarray,
    alpha: float,
    gap_threshold: float,
) -> np.ndarray:
    pred = shared_logits.argmax(axis=1).astype(np.int64)
    baseline_task = block_scores.argmax(axis=1)
    for i in range(shared_logits.shape[0]):
        candidate_tasks = candidate_tasks_by_k[i]
        if candidate_tasks.size <= 1:
            continue
        candidate_block = block_scores[i, candidate_tasks]
        order = np.argsort(-candidate_block)
        gap = float(candidate_block[order[0]] - candidate_block[order[1]])
        if gap >= gap_threshold:
            continue
        score_vec = block_scores[i].copy()
        score_vec[candidate_tasks] += float(alpha) * metric[i, candidate_tasks]
        chosen_task = choose_task_by_score(candidate_tasks, score_vec, int(baseline_task[i]))
        start, end = int(task_starts[chosen_task]), int(task_ends[chosen_task])
        pred[i] = start + int(np.argmax(shared_logits[i, start:end]))
    return pred


def grid_values(preset: str) -> Dict[str, Sequence[Any]]:
    if preset == "quick":
        return {
            "candidate_topk": [2, 3],
            "rerank_topk": [3, 5],
            "temperature": [0.5, 1.0, 2.0],
            "alpha_scale": [0.1, 0.2, 0.5],
            "alpha_bonus": [0.05, 0.1, 0.2, 0.5],
            "gap_quantile": [0.2, 0.4],
        }
    if preset == "full":
        return {
            "candidate_topk": [2, 3, 5],
            "rerank_topk": [2, 3, 5],
            "temperature": [0.25, 0.5, 1.0, 2.0, 4.0],
            "alpha_scale": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            "alpha_bonus": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            "gap_quantile": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    return {
        "candidate_topk": [2, 3, 5],
        "rerank_topk": [3, 5],
        "temperature": [0.5, 1.0, 2.0, 4.0],
        "alpha_scale": [0.05, 0.1, 0.2, 0.5, 1.0],
        "alpha_bonus": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        "gap_quantile": [0.1, 0.2, 0.3, 0.4],
    }


def run_sweep(
    shared_logits: np.ndarray,
    targets: np.ndarray,
    task_scores: np.ndarray,
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    class_to_task: np.ndarray,
    metrics: Dict[str, Any],
    candidate_topk_values: Sequence[int],
    rerank_topk_values: Sequence[int],
    preset: str,
    output_dir: Path,
) -> None:
    max_needed_topk = max(5, max(candidate_topk_values), max(rerank_topk_values))
    shared_topk = rank_desc(shared_logits, topk=max_needed_topk)
    baseline_pred = method_baseline_shared(shared_logits)
    baseline_acc = top1_accuracy(baseline_pred, targets)
    block_variants = {
        "block_lse": metrics["block_lse"],
        "block_max": metrics["block_max"],
    }
    metric_bank = metrics["metric_bank"]
    candidate_tasks_lookup = {
        k: candidate_tasks_from_topk_classes(shared_topk, class_to_task, k) for k in candidate_topk_values
    }

    gt_task = class_to_task[targets]
    oracle_shared_pred = method_oracle_shared_local(targets, class_to_task, metrics["shared_local_top1"])

    oracle_summary = {
        "baseline_shared_top1": baseline_acc,
        "oracle_shared_with_gt_task_top1": top1_accuracy(oracle_shared_pred, targets),
        "detector_task_top1_raw": float((task_scores.argmax(axis=1) == gt_task).mean() * 100.0),
        "gt_task_in_shared_top2_tasks": float(
            np.mean(
                [gt_task[i] in candidate_tasks_lookup[min(2, max(candidate_topk_values))][i] for i in range(targets.shape[0])]
            )
            * 100.0
        ),
        "gt_task_in_shared_top3_tasks": float(
            np.mean(
                [gt_task[i] in candidate_tasks_lookup[min(3, max(candidate_topk_values))][i] for i in range(targets.shape[0])]
            )
            * 100.0
        ),
    }
    for metric_name, metric in metric_bank.items():
        oracle_summary[f"route_top1/{metric_name}"] = float((metric.argmax(axis=1) == gt_task).mean() * 100.0)

    with open(output_dir / "oracle_summary.json", "w", encoding="utf-8") as f:
        json.dump(oracle_summary, f, indent=2)

    grids = grid_values(preset)
    rows: List[Dict[str, Any]] = []

    def push_row(method_name: str, pred: np.ndarray, **params: Any) -> None:
        acc = top1_accuracy(pred, targets)
        row = {
            "method": method_name,
            "top1": round(acc, 4),
            "delta_vs_baseline": round(acc - baseline_acc, 4),
        }
        row.update(grouped_correction_stats(pred, baseline_pred, targets))
        row.update(params)
        rows.append(row)

    push_row("baseline_shared", baseline_pred)
    push_row("oracle_shared_with_gt_task", oracle_shared_pred)

    for metric_name, metric in metric_bank.items():
        for scale_mode, temperature, alpha in itertools.product(
            ["mul_centered", "mul_softmax", "add_softmax"],
            grids["temperature"],
            grids["alpha_scale"],
        ):
            pred = method_softmax_scale_then_shared(
                shared_logits=shared_logits,
                metric=metric,
                task_starts=task_starts,
                task_ends=task_ends,
                alpha=float(alpha),
                temperature=float(temperature),
                mode=scale_mode,
            )
            push_row(
                "softmax_scale_then_shared",
                pred,
                metric=metric_name,
                scale_mode=scale_mode,
                temperature=temperature,
                alpha=alpha,
            )

        for alpha in grids["alpha_bonus"]:
            pred = method_task_bonus_all_classes(
                shared_logits=shared_logits,
                metric=metric,
                task_starts=task_starts,
                task_ends=task_ends,
                alpha=float(alpha),
            )
            push_row(
                "task_bonus_all_classes",
                pred,
                metric=metric_name,
                alpha=alpha,
            )

        for candidate_topk, rerank_topk, alpha in itertools.product(
            candidate_topk_values,
            rerank_topk_values,
            grids["alpha_bonus"],
        ):
            pred = method_class_bonus_candidate_rerank(
                shared_logits=shared_logits,
                shared_topk=shared_topk,
                metric=metric,
                class_to_task=class_to_task,
                candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
                alpha=float(alpha),
                rerank_topk=int(rerank_topk),
            )
            push_row(
                "class_bonus_candidate_rerank",
                pred,
                metric=metric_name,
                candidate_topk=candidate_topk,
                rerank_topk=rerank_topk,
                alpha=alpha,
            )

        for block_name, block_scores in block_variants.items():
            for candidate_topk, alpha in itertools.product(candidate_topk_values, grids["alpha_bonus"]):
                pred = method_select_task_then_shared(
                    block_scores=block_scores,
                    metric=metric,
                    candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
                    task_starts=task_starts,
                    task_ends=task_ends,
                    shared_logits=shared_logits,
                    alpha=float(alpha),
                )
                push_row(
                    "select_task_then_shared",
                    pred,
                    metric=metric_name,
                    block=block_name,
                    candidate_topk=candidate_topk,
                    alpha=alpha,
                )

            for candidate_topk, alpha, gap_q in itertools.product(
                candidate_topk_values,
                grids["alpha_bonus"],
                grids["gap_quantile"],
            ):
                gaps = []
                for i in range(shared_logits.shape[0]):
                    candidate_tasks = candidate_tasks_lookup[candidate_topk][i]
                    if candidate_tasks.size <= 1:
                        continue
                    candidate_block = block_scores[i, candidate_tasks]
                    order = np.argsort(-candidate_block)
                    gaps.append(float(candidate_block[order[0]] - candidate_block[order[1]]))
                gap_threshold = float(np.quantile(np.array(gaps, dtype=np.float32), gap_q)) if gaps else math.inf
                pred = method_gated_select_task_then_shared(
                    block_scores=block_scores,
                    metric=metric,
                    candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
                    task_starts=task_starts,
                    task_ends=task_ends,
                    shared_logits=shared_logits,
                    alpha=float(alpha),
                    gap_threshold=gap_threshold,
                )
                push_row(
                    "gated_select_task_then_shared",
                    pred,
                    metric=metric_name,
                    block=block_name,
                    candidate_topk=candidate_topk,
                    alpha=alpha,
                    gap_quantile=gap_q,
                    gap_threshold=gap_threshold,
                )

    rows_sorted = sorted(rows, key=lambda x: (-x["top1"], x["method"]))
    all_csv = output_dir / "prior_sweep_all.csv"
    best_csv = output_dir / "prior_sweep_best_per_method.csv"
    all_fieldnames = sorted({key for row in rows_sorted for key in row.keys()})

    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    best_rows = []
    seen_keys = set()
    for row in rows_sorted:
        method_key = (row["method"], row.get("metric"), row.get("block"), row.get("scale_mode"))
        if method_key in seen_keys:
            continue
        seen_keys.add(method_key)
        best_rows.append(row)

    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)

    print("\n[summary] Oracle analysis")
    for key, value in oracle_summary.items():
        print(f"  {key}: {value}")

    print("\n[summary] Top global rows")
    for row in rows_sorted[:20]:
        metric = row.get("metric", "-")
        print(
            f"  {row['method']:<28} metric={metric:<24} top1={row['top1']:.4f} "
            f"delta={row['delta_vs_baseline']:+.4f}"
        )

    print(f"\n[done] Saved full sweep to: {all_csv}")
    print(f"[done] Saved best-per-family to: {best_csv}")
    print(f"[done] Saved oracle summary to: {output_dir / 'oracle_summary.json'}")


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    norm_args = normalize_args(cfg, args.device, args.batch_size)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if norm_args.get("model_name", "spie_v14").lower() != "spie_v14":
        print(f"[warn] model_name in config is {norm_args.get('model_name')!r}; overriding to 'spie_v14'.")
        norm_args["model_name"] = "spie_v14"

    data_manager = DataManager(
        norm_args["dataset"],
        norm_args["shuffle"],
        norm_args["seed"],
        norm_args["init_cls"],
        norm_args["increment"],
        norm_args,
    )
    norm_args["nb_classes"] = data_manager.nb_classes
    norm_args["nb_tasks"] = data_manager.nb_tasks
    learner = factory.get_model(norm_args["model_name"], norm_args)
    rebuild_spie_v14_to_checkpoint(learner, data_manager, checkpoint)
    learner._network.to(norm_args["device"][0])
    learner._network.eval()

    loader = build_test_loader(
        data_manager=data_manager,
        total_classes=learner._total_classes,
        batch_size=norm_args["batch_size"],
        num_workers=args.num_workers,
    )

    output_dir, cache_file = cache_paths(args.output_dir, args.cache_name)
    cache_metadata = build_cache_metadata(args.config, args.checkpoint, norm_args, learner)
    cache = extract_or_load_scores(learner, loader, args.force_recache, cache_file, cache_metadata)

    class_to_task = build_class_to_task(cache["task_starts"], cache["task_ends"])
    metrics = precompute_metrics(
        shared_logits=cache["shared_logits"],
        task_scores=cache["task_scores"],
        task_stats=cache["task_stats"],
        task_starts=cache["task_starts"],
        task_ends=cache["task_ends"],
    )

    grid = grid_values(args.grid_preset)
    candidate_topk_values = args.candidate_task_from_topk or list(grid["candidate_topk"])
    rerank_topk_values = args.class_rerank_topk or list(grid["rerank_topk"])

    print("[info] Sweep settings")
    print("  checkpoint:", args.checkpoint)
    print("  output_dir:", output_dir)
    print("  total_classes:", learner._total_classes)
    print("  num_tasks:", len(learner.task_class_ranges))
    print("  candidate_topk:", candidate_topk_values)
    print("  rerank_topk:", rerank_topk_values)
    print("  grid_preset:", args.grid_preset)

    run_sweep(
        shared_logits=cache["shared_logits"],
        targets=cache["targets"],
        task_scores=cache["task_scores"],
        task_starts=cache["task_starts"],
        task_ends=cache["task_ends"],
        class_to_task=class_to_task,
        metrics=metrics,
        candidate_topk_values=candidate_topk_values,
        rerank_topk_values=rerank_topk_values,
        preset=args.grid_preset,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
