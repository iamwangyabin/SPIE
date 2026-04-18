#!/usr/bin/env python3
"""
Post-hoc fusion sweep for SPIE v17.

What it does
------------
1. Loads a finished SPIE v17 checkpoint.
2. Rebuilds the shared head + expert heads to match the checkpoint.
3. Runs the test set once and caches:
   - shared global logits
   - every expert head's local logits
4. Sweeps a broad family of post-hoc fusion rules without retraining.
5. Saves CSV summaries sorted by accuracy.

This script intentionally treats the test set as an oracle tuning playground.
Use it for analysis / upper-bound exploration, not as a clean final protocol.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPIE v17 post-hoc fusion sweep")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config used for training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to task_*.pkl checkpoint.")
    parser.add_argument("--output-dir", type=str, default="fusion_sweep_v17_out", help="Directory to save cache + CSVs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for extraction.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers for test extraction.")
    parser.add_argument("--force-recache", action="store_true", help="Recompute cached logits even if cache exists.")
    parser.add_argument(
        "--cache-name",
        type=str,
        default="cached_logits_v17.npz",
        help="Filename for cached shared/expert logits under output-dir.",
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


def logsumexp_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    stable = x - x_max
    out = x_max + np.log(np.sum(np.exp(stable), axis=axis, keepdims=True) + EPS)
    return np.squeeze(out, axis=axis)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True).clip(min=EPS)


def entropy_from_logits(x: np.ndarray, axis: int = -1) -> np.ndarray:
    p = softmax_np(x, axis=axis)
    return -np.sum(p * np.log(p.clip(min=EPS)), axis=axis)


def top1_top2_gap(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.squeeze(x[:, :1], axis=1)
    part = np.partition(x, kth=x.shape[1] - 2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    return top1 - top2


def centered_cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = a - a.mean(axis=1, keepdims=True)
    b0 = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a0 * b0, axis=1)
    den = np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)
    return num / np.clip(den, a_min=EPS, a_max=None)


def rank_desc(x: np.ndarray, topk: int) -> np.ndarray:
    topk = min(topk, x.shape[1])
    order = np.argsort(-x, axis=1)
    return order[:, :topk]


def zscore_rows_np(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.clip(std, a_min=EPS, a_max=None)


def minmax_rows_np(x: np.ndarray) -> np.ndarray:
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    return (x - x_min) / np.clip(x_max - x_min, a_min=EPS, a_max=None)


def normalize_rows_np(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "zscore":
        return zscore_rows_np(x)
    if mode == "minmax":
        return minmax_rows_np(x)
    raise ValueError(f"Unknown normalize mode: {mode}")


def rebuild_spie_v17_to_checkpoint(learner, data_manager: DataManager, checkpoint: Dict[str, Any]) -> None:
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
        learner._network.append_expert_head(task_size)
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


def extract_or_load_logits(
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
            num_tasks = int(data["num_tasks"])
            expert_logits = [data[f"expert_logits_{i}"] for i in range(num_tasks)]
            return {
                "shared_logits": data["shared_logits"],
                "targets": data["targets"],
                "task_starts": data["task_starts"],
                "task_ends": data["task_ends"],
                "expert_logits": expert_logits,
                "num_tasks": num_tasks,
                "total_classes": int(data["total_classes"]),
            }

    print("[info] Extracting shared/expert logits from the test set...")
    learner._network.eval()
    device = learner._device
    task_ids = list(range(len(learner.task_class_ranges)))
    shared_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    expert_chunks: List[List[np.ndarray]] = [[] for _ in task_ids]

    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            shared_logits = learner._shared_cls_logits(inputs)
            expert_logits_map = learner._collect_expert_logits(inputs, task_ids)

            shared_chunks.append(shared_logits.detach().cpu().numpy().astype(np.float32))
            target_chunks.append(targets.numpy().astype(np.int64))
            for task_id in task_ids:
                expert_chunks[task_id].append(expert_logits_map[task_id].detach().cpu().numpy().astype(np.float32))

    shared_logits_np = np.concatenate(shared_chunks, axis=0)
    targets_np = np.concatenate(target_chunks, axis=0)
    expert_logits_np = [np.concatenate(chunks, axis=0) for chunks in expert_chunks]
    task_starts = np.array([start for start, _ in learner.task_class_ranges], dtype=np.int64)
    task_ends = np.array([end for _, end in learner.task_class_ranges], dtype=np.int64)

    payload = {
        "shared_logits": shared_logits_np,
        "targets": targets_np,
        "task_starts": task_starts,
        "task_ends": task_ends,
        "num_tasks": np.int64(len(task_ids)),
        "total_classes": np.int64(learner._total_classes),
        "cache_metadata": np.array(cache_metadata, dtype=object),
    }
    for task_id, arr in enumerate(expert_logits_np):
        payload[f"expert_logits_{task_id}"] = arr
    np.savez_compressed(cache_file, **payload)
    print(f"[info] Saved cache: {cache_file}")

    return {
        "shared_logits": shared_logits_np,
        "targets": targets_np,
        "task_starts": task_starts,
        "task_ends": task_ends,
        "expert_logits": expert_logits_np,
        "num_tasks": len(task_ids),
        "total_classes": learner._total_classes,
    }


def build_class_to_task(task_starts: np.ndarray, task_ends: np.ndarray) -> np.ndarray:
    total_classes = int(task_ends[-1])
    arr = np.full((total_classes,), -1, dtype=np.int64)
    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        arr[start:end] = task_id
    return arr


def support_union_js_similarity(shared_slice: np.ndarray, expert_slice: np.ndarray, local_topk: int) -> np.ndarray:
    n = shared_slice.shape[0]
    out = np.zeros((n,), dtype=np.float32)
    k_s = min(local_topk, shared_slice.shape[1])
    k_e = min(local_topk, expert_slice.shape[1])

    for i in range(n):
        s_idx = np.argpartition(-shared_slice[i], kth=k_s - 1)[:k_s]
        e_idx = np.argpartition(-expert_slice[i], kth=k_e - 1)[:k_e]
        support = np.unique(np.concatenate([s_idx, e_idx]))
        s_prob = softmax_np(shared_slice[i, support][None, :], axis=1)[0]
        e_prob = softmax_np(expert_slice[i, support][None, :], axis=1)[0]
        midpoint = 0.5 * (s_prob + e_prob)
        s_kl = np.sum(s_prob * (np.log(s_prob.clip(min=EPS)) - np.log(midpoint.clip(min=EPS))))
        e_kl = np.sum(e_prob * (np.log(e_prob.clip(min=EPS)) - np.log(midpoint.clip(min=EPS))))
        js = 0.5 * (s_kl + e_kl)
        out[i] = float(np.exp(-js))
    return out


def precompute_metrics(
    shared_logits: np.ndarray,
    expert_logits: Sequence[np.ndarray],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    local_topk: int,
) -> Dict[str, Any]:
    n = shared_logits.shape[0]
    num_tasks = len(expert_logits)
    block_scores = np.zeros((n, num_tasks), dtype=np.float32)
    sim_js = np.zeros((n, num_tasks), dtype=np.float32)
    sim_cos = np.zeros((n, num_tasks), dtype=np.float32)
    sim_cos_z = np.zeros((n, num_tasks), dtype=np.float32)
    expert_margin = np.zeros((n, num_tasks), dtype=np.float32)
    expert_neg_entropy = np.zeros((n, num_tasks), dtype=np.float32)
    shared_local_top1 = np.zeros((n, num_tasks), dtype=np.int64)
    expert_local_top1 = np.zeros((n, num_tasks), dtype=np.int64)

    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        s = shared_logits[:, start:end]
        e = expert_logits[task_id]
        block_scores[:, task_id] = logsumexp_np(s, axis=1)
        sim_cos[:, task_id] = centered_cosine_batch(s, e).astype(np.float32)
        sim_cos_z[:, task_id] = centered_cosine_batch(zscore_rows_np(s), zscore_rows_np(e)).astype(np.float32)
        sim_js[:, task_id] = support_union_js_similarity(s, e, local_topk=local_topk)
        expert_margin[:, task_id] = top1_top2_gap(e).astype(np.float32)
        expert_neg_entropy[:, task_id] = (-entropy_from_logits(e, axis=1)).astype(np.float32)
        shared_local_top1[:, task_id] = s.argmax(axis=1) + start
        expert_local_top1[:, task_id] = e.argmax(axis=1) + start

    return {
        "block_scores": block_scores,
        "sim_js": sim_js,
        "sim_cos": sim_cos,
        "sim_cos_z": sim_cos_z,
        "expert_margin": expert_margin,
        "expert_neg_entropy": expert_neg_entropy,
        "shared_local_top1": shared_local_top1,
        "expert_local_top1": expert_local_top1,
    }


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


def method_baseline_shared(shared_logits: np.ndarray) -> np.ndarray:
    return shared_logits.argmax(axis=1).astype(np.int64)


def method_oracle_shared_local(targets: np.ndarray, class_to_task: np.ndarray, shared_local_top1: np.ndarray) -> np.ndarray:
    gt_task = class_to_task[targets]
    return shared_local_top1[np.arange(targets.shape[0]), gt_task]


def method_oracle_expert_local(targets: np.ndarray, class_to_task: np.ndarray, expert_local_top1: np.ndarray) -> np.ndarray:
    gt_task = class_to_task[targets]
    return expert_local_top1[np.arange(targets.shape[0]), gt_task]


def method_task_relation_all_classes(
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


def method_class_bonus(
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
                scores[j] += float(alpha) * float(metric[i, task_id])
        pred[i] = int(candidate_classes[int(np.argmax(scores))])
    return pred


def method_class_local_bonus(
    shared_logits: np.ndarray,
    shared_topk: np.ndarray,
    expert_logits: Sequence[np.ndarray],
    class_to_task: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    task_starts: np.ndarray,
    beta: float,
    rerank_topk: int,
    normalize_mode: str,
) -> np.ndarray:
    pred = shared_logits.argmax(axis=1).astype(np.int64)
    width = min(rerank_topk, shared_topk.shape[1])
    expert_lookup = [normalize_rows_np(arr, normalize_mode) for arr in expert_logits]
    for i in range(shared_logits.shape[0]):
        candidate_classes = shared_topk[i, :width]
        scores = shared_logits[i, candidate_classes].copy()
        candidate_task_set = set(candidate_tasks_by_k[i].tolist())
        for j, class_idx in enumerate(candidate_classes.tolist()):
            task_id = int(class_to_task[class_idx])
            if task_id not in candidate_task_set:
                continue
            local_idx = int(class_idx - task_starts[task_id])
            scores[j] += float(beta) * float(expert_lookup[task_id][i, local_idx])
        pred[i] = int(candidate_classes[int(np.argmax(scores))])
    return pred


def method_local_bonus_all_classes(
    shared_logits: np.ndarray,
    expert_logits: Sequence[np.ndarray],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    beta: float,
    normalize_mode: str,
) -> np.ndarray:
    adjusted = shared_logits.copy()
    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        adjusted[:, start:end] += float(beta) * normalize_rows_np(expert_logits[task_id], normalize_mode)
    return adjusted.argmax(axis=1).astype(np.int64)


def method_task_bonus_select_task_then_shared(
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


def method_select_task_then_local_bonus(
    block_scores: np.ndarray,
    metric: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    shared_logits: np.ndarray,
    expert_logits: Sequence[np.ndarray],
    alpha: float,
    beta: float,
    normalize_mode: str,
) -> np.ndarray:
    pred = np.zeros((shared_logits.shape[0],), dtype=np.int64)
    baseline_task = block_scores.argmax(axis=1)
    expert_lookup = [normalize_rows_np(arr, normalize_mode) for arr in expert_logits]
    for i in range(shared_logits.shape[0]):
        candidate_tasks = candidate_tasks_by_k[i]
        if candidate_tasks.size == 0:
            chosen_task = int(baseline_task[i])
        else:
            score_vec = block_scores[i].copy()
            score_vec[candidate_tasks] += float(alpha) * metric[i, candidate_tasks]
            chosen_task = choose_task_by_score(candidate_tasks, score_vec, int(baseline_task[i]))
        start, end = int(task_starts[chosen_task]), int(task_ends[chosen_task])
        local_scores = shared_logits[i, start:end] + float(beta) * expert_lookup[chosen_task][i]
        pred[i] = start + int(np.argmax(local_scores))
    return pred


def method_task_bonus_select_task_then_expert(
    block_scores: np.ndarray,
    metric: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    expert_local_top1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    pred = np.zeros((block_scores.shape[0],), dtype=np.int64)
    baseline_task = block_scores.argmax(axis=1)
    for i in range(block_scores.shape[0]):
        candidate_tasks = candidate_tasks_by_k[i]
        if candidate_tasks.size == 0:
            chosen_task = int(baseline_task[i])
        else:
            score_vec = block_scores[i].copy()
            score_vec[candidate_tasks] += float(alpha) * metric[i, candidate_tasks]
            chosen_task = choose_task_by_score(candidate_tasks, score_vec, int(baseline_task[i]))
        pred[i] = int(expert_local_top1[i, chosen_task])
    return pred


def method_task_bonus_combo_then_shared(
    block_scores: np.ndarray,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    shared_logits: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    pred = np.zeros((shared_logits.shape[0],), dtype=np.int64)
    baseline_task = block_scores.argmax(axis=1)
    for i in range(shared_logits.shape[0]):
        candidate_tasks = candidate_tasks_by_k[i]
        if candidate_tasks.size == 0:
            chosen_task = int(baseline_task[i])
        else:
            score_vec = block_scores[i].copy()
            score_vec[candidate_tasks] += float(alpha) * metric_a[i, candidate_tasks] + float(beta) * metric_b[
                i, candidate_tasks
            ]
            chosen_task = choose_task_by_score(candidate_tasks, score_vec, int(baseline_task[i]))
        start, end = int(task_starts[chosen_task]), int(task_ends[chosen_task])
        pred[i] = start + int(np.argmax(shared_logits[i, start:end]))
    return pred


def method_gated_task_bonus_then_shared(
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
            "alpha_relation": [0.05, 0.1, 0.2, 0.5, 1.0],
            "beta_local": [0.05, 0.1, 0.2, 0.5],
            "alpha_margin": [0.05, 0.1, 0.2],
            "alpha_entropy": [0.05, 0.1, 0.2],
            "beta_margin": [0.05, 0.1],
            "normalize_mode": ["none", "zscore"],
            "gap_quantile": [0.2, 0.4],
        }
    if preset == "full":
        return {
            "candidate_topk": [2, 3, 5],
            "rerank_topk": [2, 3, 5],
            "temperature": [0.25, 0.5, 1.0, 2.0, 4.0],
            "alpha_relation": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "beta_local": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
            "alpha_margin": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
            "alpha_entropy": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
            "beta_margin": [0.01, 0.03, 0.05, 0.1, 0.2, 0.5],
            "normalize_mode": ["none", "zscore", "minmax"],
            "gap_quantile": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    return {
        "candidate_topk": [2, 3, 5],
        "rerank_topk": [3, 5],
        "temperature": [0.5, 1.0, 2.0, 4.0],
        "alpha_relation": [0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "beta_local": [0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
        "alpha_margin": [0.01, 0.05, 0.1, 0.2, 0.5],
        "alpha_entropy": [0.01, 0.05, 0.1, 0.2, 0.5],
        "beta_margin": [0.01, 0.05, 0.1, 0.2],
        "normalize_mode": ["none", "zscore", "minmax"],
        "gap_quantile": [0.1, 0.2, 0.3, 0.4, 0.5],
    }


def run_sweep(
    shared_logits: np.ndarray,
    targets: np.ndarray,
    task_starts: np.ndarray,
    task_ends: np.ndarray,
    class_to_task: np.ndarray,
    metrics: Dict[str, Any],
    candidate_topk_values: Sequence[int],
    rerank_topk_values: Sequence[int],
    preset: str,
    output_dir: Path,
) -> None:
    shared_topk = rank_desc(shared_logits, topk=max(5, max(rerank_topk_values)))
    block_scores = metrics["block_scores"]
    baseline_pred = method_baseline_shared(shared_logits)
    baseline_acc = top1_accuracy(baseline_pred, targets)

    candidate_tasks_lookup = {
        k: candidate_tasks_from_topk_classes(shared_topk, class_to_task, k) for k in candidate_topk_values
    }

    gt_task = class_to_task[targets]
    oracle_shared_pred = method_oracle_shared_local(targets, class_to_task, metrics["shared_local_top1"])
    oracle_expert_pred = method_oracle_expert_local(targets, class_to_task, metrics["expert_local_top1"])
    oracle_summary = {
        "baseline_shared_top1": baseline_acc,
        "oracle_shared_with_gt_task_top1": top1_accuracy(oracle_shared_pred, targets),
        "oracle_expert_with_gt_task_top1": top1_accuracy(oracle_expert_pred, targets),
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
        "mean_js_true_task": float(metrics["sim_js"][np.arange(targets.shape[0]), gt_task].mean()),
        "mean_cos_true_task": float(metrics["sim_cos"][np.arange(targets.shape[0]), gt_task].mean()),
        "mean_cos_z_true_task": float(metrics["sim_cos_z"][np.arange(targets.shape[0]), gt_task].mean()),
    }
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

    for metric_name in ["sim_cos", "sim_cos_z", "sim_js"]:
        for alpha in grids["alpha_relation"]:
            pred = method_task_relation_all_classes(
                shared_logits=shared_logits,
                metric=metrics[metric_name],
                task_starts=task_starts,
                task_ends=task_ends,
                alpha=float(alpha),
            )
            push_row("task_relation_all_classes", pred, metric=metric_name, alpha=alpha)

    for metric_name in ["sim_cos", "sim_cos_z", "sim_js"]:
        for scale_mode, temperature, alpha in itertools.product(
            ["mul_centered", "mul_softmax", "add_softmax"],
            grids["temperature"],
            grids["alpha_relation"],
        ):
            pred = method_softmax_scale_then_shared(
                shared_logits=shared_logits,
                metric=metrics[metric_name],
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

    for normalize_mode, beta in itertools.product(grids["normalize_mode"], grids["beta_local"]):
        pred = method_local_bonus_all_classes(
            shared_logits=shared_logits,
            expert_logits=expert_logits,
            task_starts=task_starts,
            task_ends=task_ends,
            beta=float(beta),
            normalize_mode=normalize_mode,
        )
        push_row(
            "local_bonus_all_classes",
            pred,
            normalize_mode=normalize_mode,
            beta=beta,
        )

    for metric_name in ["sim_js", "sim_cos", "sim_cos_z"]:
        for candidate_topk, rerank_topk, alpha in itertools.product(
            candidate_topk_values, rerank_topk_values, grids["alpha_relation"]
        ):
            pred = method_class_bonus(
                shared_logits=shared_logits,
                shared_topk=shared_topk,
                metric=metrics[metric_name],
                class_to_task=class_to_task,
                candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
                alpha=float(alpha),
                rerank_topk=int(rerank_topk),
            )
            push_row(
                "class_bonus",
                pred,
                metric=metric_name,
                candidate_topk=candidate_topk,
                rerank_topk=rerank_topk,
                alpha=alpha,
            )

    for candidate_topk, rerank_topk, normalize_mode, beta in itertools.product(
        candidate_topk_values,
        rerank_topk_values,
        grids["normalize_mode"],
        grids["beta_local"],
    ):
        pred = method_class_local_bonus(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            expert_logits=expert_logits,
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            task_starts=task_starts,
            beta=float(beta),
            rerank_topk=int(rerank_topk),
            normalize_mode=normalize_mode,
        )
        push_row(
            "class_local_bonus",
            pred,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            normalize_mode=normalize_mode,
            beta=beta,
        )

    for metric_name in ["sim_js", "sim_cos", "sim_cos_z", "expert_margin", "expert_neg_entropy"]:
        alpha_grid = grids["alpha_relation"] if metric_name.startswith("sim_") else (
            grids["alpha_margin"] if metric_name == "expert_margin" else grids["alpha_entropy"]
        )
        for candidate_topk, alpha in itertools.product(candidate_topk_values, alpha_grid):
            pred = method_task_bonus_select_task_then_shared(
                block_scores=block_scores,
                metric=metrics[metric_name],
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
                candidate_topk=candidate_topk,
                alpha=alpha,
            )

    for metric_name in ["sim_js", "sim_cos", "sim_cos_z", "expert_margin", "expert_neg_entropy"]:
        alpha_grid = grids["alpha_relation"] if metric_name.startswith("sim_") else (
            grids["alpha_margin"] if metric_name == "expert_margin" else grids["alpha_entropy"]
        )
        for candidate_topk, normalize_mode, alpha, beta in itertools.product(
            candidate_topk_values,
            grids["normalize_mode"],
            alpha_grid,
            grids["beta_local"],
        ):
            pred = method_select_task_then_local_bonus(
                block_scores=block_scores,
                metric=metrics[metric_name],
                candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
                task_starts=task_starts,
                task_ends=task_ends,
                shared_logits=shared_logits,
                expert_logits=expert_logits,
                alpha=float(alpha),
                beta=float(beta),
                normalize_mode=normalize_mode,
            )
            push_row(
                "select_task_then_local_bonus",
                pred,
                metric=metric_name,
                candidate_topk=candidate_topk,
                normalize_mode=normalize_mode,
                alpha=alpha,
                beta=beta,
            )

    for candidate_topk, alpha in itertools.product(candidate_topk_values, grids["alpha_relation"]):
        pred = method_task_bonus_select_task_then_expert(
            block_scores=block_scores,
            metric=metrics["sim_js"],
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            expert_local_top1=metrics["expert_local_top1"],
            alpha=float(alpha),
        )
        push_row("select_task_then_expert", pred, metric="sim_js", candidate_topk=candidate_topk, alpha=alpha)

    for metric_name in ["sim_js", "sim_cos", "sim_cos_z"]:
        for candidate_topk, alpha, beta in itertools.product(
            candidate_topk_values, grids["alpha_relation"], grids["beta_margin"]
        ):
            pred = method_task_bonus_combo_then_shared(
                block_scores=block_scores,
                metric_a=metrics[metric_name],
                metric_b=metrics["expert_margin"],
                candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
                task_starts=task_starts,
                task_ends=task_ends,
                shared_logits=shared_logits,
                alpha=float(alpha),
                beta=float(beta),
            )
            push_row(
                "select_task_combo_then_shared",
                pred,
                metric_a=metric_name,
                metric_b="expert_margin",
                candidate_topk=candidate_topk,
                alpha=alpha,
                beta=beta,
            )

    for metric_name in ["sim_js", "sim_cos", "sim_cos_z"]:
        for candidate_topk, alpha, gap_q in itertools.product(
            candidate_topk_values, grids["alpha_relation"], grids["gap_quantile"]
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
            pred = method_gated_task_bonus_then_shared(
                block_scores=block_scores,
                metric=metrics[metric_name],
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
                candidate_topk=candidate_topk,
                alpha=alpha,
                gap_quantile=gap_q,
                gap_threshold=gap_threshold,
            )

    rows_sorted = sorted(rows, key=lambda x: (-x["top1"], x["method"]))
    all_csv = output_dir / "fusion_sweep_all.csv"
    best_csv = output_dir / "fusion_sweep_best_per_method.csv"
    all_fieldnames = sorted({key for row in rows_sorted for key in row.keys()})

    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    best_rows = []
    seen_keys = set()
    for row in rows_sorted:
        key = (
            row["method"],
            row.get("metric"),
            row.get("metric_a"),
            row.get("metric_b"),
            row.get("normalize_mode"),
            row.get("scale_mode"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        best_rows.append(row)

    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)

    print("\n[summary] Oracle analysis")
    for k, v in oracle_summary.items():
        print(f"  {k}: {v}")

    print("\n[summary] Top global rows")
    for row in rows_sorted[:20]:
        metric = row.get("metric", row.get("metric_a", "-"))
        print(
            f"  {row['method']:<30} metric={metric:<16} top1={row['top1']:.4f} "
            f"delta={row['delta_vs_baseline']:+.4f}"
        )

    print(f"\n[done] Saved full sweep to: {all_csv}")
    print(f"[done] Saved best-per-method to: {best_csv}")
    print(f"[done] Saved oracle summary to: {output_dir / 'oracle_summary.json'}")


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    norm_args = normalize_args(cfg, args.device, args.batch_size)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if norm_args.get("model_name", "spie_v17").lower() != "spie_v17":
        print(f"[warn] model_name in config is {norm_args.get('model_name')!r}; overriding to 'spie_v17'.")
        norm_args["model_name"] = "spie_v17"

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
    rebuild_spie_v17_to_checkpoint(learner, data_manager, checkpoint)
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
    cache = extract_or_load_logits(learner, loader, args.force_recache, cache_file, cache_metadata)

    class_to_task = build_class_to_task(cache["task_starts"], cache["task_ends"])
    metrics = precompute_metrics(
        shared_logits=cache["shared_logits"],
        expert_logits=cache["expert_logits"],
        task_starts=cache["task_starts"],
        task_ends=cache["task_ends"],
        local_topk=int(norm_args.get("relation_local_topk", 3)),
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
