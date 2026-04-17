#!/usr/bin/env python3
"""
Deeper post-hoc sweep around class_bonus_cos for SPIE v16.

Focus:
- reuse cached shared/expert logits when available
- explore safer cosine-based fusion families without retraining
- sweep more conservative / class-aware / gated variants

Typical usage from repo root:
python spie_v16_cosine_deep_sweep.py \
  --config exps/your_spie_v16.json \
  --checkpoint logs/.../checkpoints/task_9.pkl \
  --output-dir sweep_spie_v16_cos_deep \
  --base-cache sweep_spie_v16/cached_logits.npz \
  --grid-preset medium
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import factory
from utils.data_manager import DataManager

EPS = 1e-12

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deeper cosine-heavy post-hoc sweep for SPIE v16")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="fusion_sweep_cos_deep")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force-recache", action="store_true")
    parser.add_argument("--cache-name", type=str, default="cached_logits.npz")
    parser.add_argument(
        "--base-cache",
        type=str,
        default=None,
        help="Optional path to an existing cached_logits.npz to reuse directly.",
    )
    parser.add_argument("--grid-preset", type=str, default="medium", choices=["quick", "medium", "full"])
    parser.add_argument("--candidate-task-from-topk", type=int, nargs="*", default=None)
    parser.add_argument("--class-rerank-topk", type=int, nargs="*", default=None)
    parser.add_argument("--max-shared-topk", type=int, default=20)
    parser.add_argument("--progress", action="store_true", help="Show progress bars when tqdm is available.")
    return parser.parse_args()


def maybe_tqdm(iterable, *, total: int | None = None, desc: str = "", enabled: bool = False):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def progress_log(enabled: bool, current: int, total: int, desc: str) -> None:
    if not enabled or tqdm is not None or total <= 0:
        return
    checkpoints = {1, total}
    for ratio in (0.1, 0.25, 0.5, 0.75, 0.9):
        checkpoints.add(max(1, min(total, int(total * ratio))))
    if current in checkpoints:
        print(f"[progress] {desc}: {current}/{total} ({100.0 * current / total:.1f}%)")


def logsumexp_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True) + EPS), axis=axis)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.clip(np.sum(ex, axis=axis, keepdims=True), a_min=EPS, a_max=None)


def entropy_from_logits(x: np.ndarray, axis: int = -1) -> np.ndarray:
    p = softmax_np(x, axis=axis)
    return -np.sum(p * np.log(np.clip(p, EPS, None)), axis=axis)


def top1_top2_gap(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.squeeze(x[:, :1], axis=1)
    part = np.partition(x, kth=x.shape[1] - 2, axis=1)
    return part[:, -1] - part[:, -2]


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / np.clip(std, EPS, None)


def centered_rows(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def centered_cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = centered_rows(a)
    b0 = centered_rows(b)
    num = np.sum(a0 * b0, axis=1)
    den = np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)
    return num / np.clip(den, EPS, None)


def zscore_cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a0 = zscore_rows(a)
    b0 = zscore_rows(b)
    num = np.sum(a0 * b0, axis=1)
    den = np.linalg.norm(a0, axis=1) * np.linalg.norm(b0, axis=1)
    return num / np.clip(den, EPS, None)


def rank_transform_desc(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-x, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    row_ids = np.arange(x.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(x.shape[1], dtype=np.float32)[None, :]
    return -ranks


def rank_cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return centered_cosine_batch(rank_transform_desc(a), rank_transform_desc(b))


def masked_topm_cosine_batch(a: np.ndarray, b: np.ndarray, topm: int) -> np.ndarray:
    n, d = a.shape
    out = np.zeros((n,), dtype=np.float32)
    m = min(topm, d)
    for i in range(n):
        a_idx = np.argpartition(-a[i], kth=m - 1)[:m]
        b_idx = np.argpartition(-b[i], kth=m - 1)[:m]
        support = np.unique(np.concatenate([a_idx, b_idx]))
        aa = a[i, support][None, :]
        bb = b[i, support][None, :]
        out[i] = float(centered_cosine_batch(aa, bb)[0])
    return out


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
        s_kl = np.sum(s_prob * (np.log(np.clip(s_prob, EPS, None)) - np.log(np.clip(midpoint, EPS, None))))
        e_kl = np.sum(e_prob * (np.log(np.clip(e_prob, EPS, None)) - np.log(np.clip(midpoint, EPS, None))))
        js = 0.5 * (s_kl + e_kl)
        out[i] = float(np.exp(-js))
    return out


def rank_desc(x: np.ndarray, topk: int) -> np.ndarray:
    topk = min(topk, x.shape[1])
    order = np.argsort(-x, axis=1)
    return order[:, :topk]


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
    args.setdefault("memory_size", 0)
    args.setdefault("fixed_memory", False)
    args.setdefault("memory_per_class", None)
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


def rebuild_spie_v16_to_checkpoint(learner, data_manager: DataManager, checkpoint: Dict[str, Any]) -> None:
    if "tasks" not in checkpoint:
        raise KeyError("Checkpoint is missing 'tasks'.")
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
    missing, unexpected = learner._network.load_state_dict(checkpoint["model_state_dict"], strict=False)
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


def _cache_dict_from_npz(data) -> Dict[str, Any]:
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


def extract_or_load_logits(
    learner,
    loader: DataLoader,
    force_recache: bool,
    cache_file: Path,
    base_cache: str | None,
    show_progress: bool,
) -> Dict[str, Any]:
    if base_cache is not None:
        base_path = Path(base_cache)
        if not base_path.exists():
            raise FileNotFoundError(f"base cache not found: {base_path}")
        print(f"[info] Loading base cache: {base_path}")
        data = np.load(base_path, allow_pickle=True)
        cache = _cache_dict_from_npz(data)
        if cache_file != base_path:
            payload = {
                "shared_logits": cache["shared_logits"],
                "targets": cache["targets"],
                "task_starts": cache["task_starts"],
                "task_ends": cache["task_ends"],
                "num_tasks": np.int64(cache["num_tasks"]),
                "total_classes": np.int64(cache["total_classes"]),
            }
            for task_id, arr in enumerate(cache["expert_logits"]):
                payload[f"expert_logits_{task_id}"] = arr
            np.savez_compressed(cache_file, **payload)
            print(f"[info] Copied base cache to: {cache_file}")
        return cache

    if cache_file.exists() and not force_recache:
        print(f"[info] Loading cache: {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return _cache_dict_from_npz(data)

    print("[info] Extracting shared/expert logits from the test set...")
    learner._network.eval()
    device = learner._device
    task_ids = list(range(len(learner.task_class_ranges)))
    shared_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    expert_chunks: List[List[np.ndarray]] = [[] for _ in task_ids]

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(
            maybe_tqdm(loader, total=total_batches, desc="extract_logits", enabled=show_progress), start=1
        ):
            inputs = inputs.to(device)
            shared_logits = learner._shared_cls_logits(inputs)
            expert_logits_map = learner._collect_expert_logits(inputs, task_ids)
            shared_chunks.append(shared_logits.detach().cpu().numpy().astype(np.float32))
            target_chunks.append(targets.numpy().astype(np.int64))
            for task_id in task_ids:
                expert_chunks[task_id].append(expert_logits_map[task_id].detach().cpu().numpy().astype(np.float32))
            if total_batches is not None:
                progress_log(show_progress, _, total_batches, "extract_logits")

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
    expert_margin = np.zeros((n, num_tasks), dtype=np.float32)
    expert_neg_entropy = np.zeros((n, num_tasks), dtype=np.float32)
    shared_local_top1 = np.zeros((n, num_tasks), dtype=np.int64)
    expert_local_top1 = np.zeros((n, num_tasks), dtype=np.int64)

    sim_center_cos = np.zeros((n, num_tasks), dtype=np.float32)
    sim_zscore_cos = np.zeros((n, num_tasks), dtype=np.float32)
    sim_rank_cos = np.zeros((n, num_tasks), dtype=np.float32)
    sim_top3_masked_cos = np.zeros((n, num_tasks), dtype=np.float32)
    sim_js = np.zeros((n, num_tasks), dtype=np.float32)

    expert_local_center: List[np.ndarray] = []
    expert_local_zscore: List[np.ndarray] = []
    expert_local_logprob: List[np.ndarray] = []

    for task_id, (start, end) in enumerate(zip(task_starts.tolist(), task_ends.tolist())):
        s = shared_logits[:, start:end]
        e = expert_logits[task_id]
        block_scores[:, task_id] = logsumexp_np(s, axis=1)
        expert_margin[:, task_id] = top1_top2_gap(e).astype(np.float32)
        expert_neg_entropy[:, task_id] = (-entropy_from_logits(e, axis=1)).astype(np.float32)
        shared_local_top1[:, task_id] = s.argmax(axis=1) + start
        expert_local_top1[:, task_id] = e.argmax(axis=1) + start

        sim_center_cos[:, task_id] = centered_cosine_batch(s, e).astype(np.float32)
        sim_zscore_cos[:, task_id] = zscore_cosine_batch(s, e).astype(np.float32)
        sim_rank_cos[:, task_id] = rank_cosine_batch(s, e).astype(np.float32)
        sim_top3_masked_cos[:, task_id] = masked_topm_cosine_batch(s, e, topm=min(3, e.shape[1])).astype(np.float32)
        sim_js[:, task_id] = support_union_js_similarity(s, e, local_topk=local_topk)

        expert_local_center.append(centered_rows(e).astype(np.float32))
        expert_local_zscore.append(zscore_rows(e).astype(np.float32))
        expert_local_logprob.append(np.log(np.clip(softmax_np(e, axis=1), EPS, None)).astype(np.float32))

    return {
        "block_scores": block_scores,
        "expert_margin": expert_margin,
        "expert_neg_entropy": expert_neg_entropy,
        "shared_local_top1": shared_local_top1,
        "expert_local_top1": expert_local_top1,
        "sim_center_cos": sim_center_cos,
        "sim_zscore_cos": sim_zscore_cos,
        "sim_rank_cos": sim_rank_cos,
        "sim_top3_masked_cos": sim_top3_masked_cos,
        "sim_js": sim_js,
        "expert_local_center": expert_local_center,
        "expert_local_zscore": expert_local_zscore,
        "expert_local_logprob": expert_local_logprob,
    }


def candidate_tasks_from_topk_classes(shared_topk: np.ndarray, class_to_task: np.ndarray, topk_width: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    width = min(topk_width, shared_topk.shape[1])
    for row in shared_topk[:, :width]:
        seen = set()
        tasks = []
        for c in row.tolist():
            t = int(class_to_task[c])
            if t not in seen:
                seen.add(t)
                tasks.append(t)
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


def candidate_tasks_filtered(candidate_tasks: np.ndarray, block_row: np.ndarray, mode: str) -> np.ndarray:
    if candidate_tasks.size <= 2:
        return candidate_tasks
    order = candidate_tasks[np.argsort(-block_row[candidate_tasks])]
    if mode == "top2_tasks_only":
        return order[:2]
    if mode == "top3_tasks_only":
        return order[:3]
    return candidate_tasks


def method_baseline_shared(shared_logits: np.ndarray) -> np.ndarray:
    return shared_logits.argmax(axis=1).astype(np.int64)


def method_oracle_shared_local(targets: np.ndarray, class_to_task: np.ndarray, shared_local_top1: np.ndarray) -> np.ndarray:
    gt_task = class_to_task[targets]
    return shared_local_top1[np.arange(targets.shape[0]), gt_task]


def method_oracle_expert_local(targets: np.ndarray, class_to_task: np.ndarray, expert_local_top1: np.ndarray) -> np.ndarray:
    gt_task = class_to_task[targets]
    return expert_local_top1[np.arange(targets.shape[0]), gt_task]


def method_class_bonus_generic(
    shared_logits: np.ndarray,
    shared_topk: np.ndarray,
    block_scores: np.ndarray,
    metric: np.ndarray,
    class_to_task: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    alpha: float,
    rerank_topk: int,
    apply_scope: str = "all_rerank_classes",
    candidate_task_mode: str = "all_candidate_tasks",
    relative_mode: str = "none",
    ref_mode: str = "top1_task",
    sign_mode: str = "both",
    gate_mode: str = "none",
    gate_values: np.ndarray | None = None,
    gate_threshold: float | None = None,
) -> np.ndarray:
    pred = shared_logits.argmax(axis=1).astype(np.int64)
    width = min(rerank_topk, shared_topk.shape[1])
    top1_tasks = class_to_task[pred]

    for i in range(shared_logits.shape[0]):
        cand_classes = shared_topk[i, :width]
        raw_tasks = candidate_tasks_by_k[i]
        cand_tasks = candidate_tasks_filtered(raw_tasks, block_scores[i], candidate_task_mode)
        if cand_tasks.size == 0:
            continue

        if gate_mode != "none":
            if gate_values is None or gate_threshold is None:
                raise ValueError("gate requested without gate values / threshold")
            if float(gate_values[i]) > float(gate_threshold):
                continue

        base_top1_task = int(top1_tasks[i])
        ref_value = 0.0
        if relative_mode != "none":
            if ref_mode == "top1_task":
                ref_value = float(metric[i, base_top1_task])
            elif ref_mode == "candidate_mean":
                ref_value = float(metric[i, cand_tasks].mean())
            elif ref_mode == "candidate_max":
                ref_value = float(metric[i, cand_tasks].max())
            else:
                raise ValueError(f"unknown ref_mode: {ref_mode}")

        scores = shared_logits[i, cand_classes].copy()
        cand_task_set = set(cand_tasks.tolist())
        for j, c in enumerate(cand_classes.tolist()):
            t = int(class_to_task[c])
            if t not in cand_task_set:
                continue
            if apply_scope == "cross_task_only" and t == base_top1_task:
                continue
            if apply_scope == "top1_top2_only" and j >= min(2, len(cand_classes)):
                continue
            bonus = float(metric[i, t])
            if relative_mode != "none":
                bonus = bonus - ref_value
            if sign_mode == "reward_only":
                bonus = max(0.0, bonus)
            elif sign_mode == "penalty_only":
                bonus = min(0.0, bonus)
            scores[j] += float(alpha) * bonus
        pred[i] = int(cand_classes[int(np.argmax(scores))])
    return pred


def method_class_bonus_expert_local(
    shared_logits: np.ndarray,
    shared_topk: np.ndarray,
    block_scores: np.ndarray,
    task_metric: np.ndarray,
    expert_local_metric: Sequence[np.ndarray],
    class_to_task: np.ndarray,
    task_starts: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    alpha_task: float,
    beta_local: float,
    rerank_topk: int,
    apply_scope: str = "all_rerank_classes",
    candidate_task_mode: str = "all_candidate_tasks",
    gate_mode: str = "none",
    gate_values: np.ndarray | None = None,
    gate_threshold: float | None = None,
    multiply_mode: str = "additive",
) -> np.ndarray:
    pred = shared_logits.argmax(axis=1).astype(np.int64)
    width = min(rerank_topk, shared_topk.shape[1])
    top1_tasks = class_to_task[pred]

    for i in range(shared_logits.shape[0]):
        cand_classes = shared_topk[i, :width]
        raw_tasks = candidate_tasks_by_k[i]
        cand_tasks = candidate_tasks_filtered(raw_tasks, block_scores[i], candidate_task_mode)
        if cand_tasks.size == 0:
            continue

        if gate_mode != "none":
            if gate_values is None or gate_threshold is None:
                raise ValueError("gate requested without gate values / threshold")
            if float(gate_values[i]) > float(gate_threshold):
                continue

        base_top1_task = int(top1_tasks[i])
        scores = shared_logits[i, cand_classes].copy()
        cand_task_set = set(cand_tasks.tolist())

        for j, c in enumerate(cand_classes.tolist()):
            t = int(class_to_task[c])
            if t not in cand_task_set:
                continue
            if apply_scope == "cross_task_only" and t == base_top1_task:
                continue
            if apply_scope == "top1_top2_only" and j >= min(2, len(cand_classes)):
                continue
            local_idx = int(c - task_starts[t])
            task_bonus = float(task_metric[i, t])
            local_bonus = float(expert_local_metric[t][i, local_idx])
            if multiply_mode == "multiplicative":
                total_bonus = alpha_task * task_bonus * local_bonus
            else:
                total_bonus = alpha_task * task_bonus + beta_local * local_bonus
            scores[j] += float(total_bonus)
        pred[i] = int(cand_classes[int(np.argmax(scores))])
    return pred


def method_class_bonus_adaptive_alpha(
    shared_logits: np.ndarray,
    shared_topk: np.ndarray,
    block_scores: np.ndarray,
    metric: np.ndarray,
    class_to_task: np.ndarray,
    candidate_tasks_by_k: List[np.ndarray],
    alpha0: float,
    rerank_topk: int,
    uncertainty: np.ndarray,
    tau: float,
    kappa: float,
    apply_scope: str,
    candidate_task_mode: str,
) -> np.ndarray:
    pred = shared_logits.argmax(axis=1).astype(np.int64)
    width = min(rerank_topk, shared_topk.shape[1])
    top1_tasks = class_to_task[pred]
    alpha_vec = alpha0 * sigmoid_np(kappa * (tau - uncertainty))

    for i in range(shared_logits.shape[0]):
        cand_classes = shared_topk[i, :width]
        raw_tasks = candidate_tasks_by_k[i]
        cand_tasks = candidate_tasks_filtered(raw_tasks, block_scores[i], candidate_task_mode)
        if cand_tasks.size == 0:
            continue
        base_top1_task = int(top1_tasks[i])
        scores = shared_logits[i, cand_classes].copy()
        cand_task_set = set(cand_tasks.tolist())
        for j, c in enumerate(cand_classes.tolist()):
            t = int(class_to_task[c])
            if t not in cand_task_set:
                continue
            if apply_scope == "cross_task_only" and t == base_top1_task:
                continue
            if apply_scope == "top1_top2_only" and j >= min(2, len(cand_classes)):
                continue
            scores[j] += float(alpha_vec[i]) * float(metric[i, t])
        pred[i] = int(cand_classes[int(np.argmax(scores))])
    return pred


def grid_values(preset: str) -> Dict[str, Sequence[Any]]:
    if preset == "quick":
        return {
            "candidate_topk": [2, 3],
            "rerank_topk": [3, 5],
            "sim_variants": ["center_cos", "zscore_cos"],
            "alpha": [0.02, 0.05, 0.1, 0.2],
            "beta_local": [0.02, 0.05, 0.1],
            "gate_quantile": [0.1, 0.2, 0.3],
            "kappa": [5.0, 10.0],
            "sign_mode": ["both", "reward_only"],
            "apply_scope": ["all_rerank_classes", "cross_task_only"],
            "candidate_task_mode": ["all_candidate_tasks", "top2_tasks_only"],
            "ref_mode": ["top1_task", "candidate_mean"],
            "local_variant": ["zscore", "logprob"],
            "multiply_mode": ["additive"],
        }
    if preset == "full":
        return {
            "candidate_topk": [2, 3, 5],
            "rerank_topk": [2, 3, 5, 10],
            "sim_variants": ["center_cos", "zscore_cos", "rank_cos", "top3_masked_cos"],
            "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            "beta_local": [0.01, 0.02, 0.05, 0.1, 0.2],
            "gate_quantile": [0.05, 0.1, 0.2, 0.3, 0.4],
            "kappa": [2.0, 5.0, 10.0, 20.0],
            "sign_mode": ["both", "reward_only", "penalty_only"],
            "apply_scope": ["all_rerank_classes", "cross_task_only", "top1_top2_only"],
            "candidate_task_mode": ["all_candidate_tasks", "top2_tasks_only", "top3_tasks_only"],
            "ref_mode": ["top1_task", "candidate_mean", "candidate_max"],
            "local_variant": ["center", "zscore", "logprob"],
            "multiply_mode": ["additive", "multiplicative"],
        }
    return {
        "candidate_topk": [2, 3, 5],
        "rerank_topk": [3, 5, 10],
        "sim_variants": ["center_cos", "zscore_cos", "rank_cos"],
        "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        "beta_local": [0.01, 0.02, 0.05, 0.1],
        "gate_quantile": [0.1, 0.2, 0.3, 0.4],
        "kappa": [5.0, 10.0, 20.0],
        "sign_mode": ["both", "reward_only", "penalty_only"],
        "apply_scope": ["all_rerank_classes", "cross_task_only", "top1_top2_only"],
        "candidate_task_mode": ["all_candidate_tasks", "top2_tasks_only"],
        "ref_mode": ["top1_task", "candidate_mean", "candidate_max"],
        "local_variant": ["zscore", "logprob"],
        "multiply_mode": ["additive", "multiplicative"],
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
    max_shared_topk: int,
    show_progress: bool,
) -> None:
    max_needed_topk = max([max_shared_topk, max(candidate_topk_values), max(rerank_topk_values), 10])
    shared_topk = rank_desc(shared_logits, topk=max_needed_topk)
    baseline_pred = method_baseline_shared(shared_logits)
    baseline_acc = top1_accuracy(baseline_pred, targets)
    block_scores = metrics["block_scores"]

    candidate_tasks_lookup = {
        k: candidate_tasks_from_topk_classes(shared_topk, class_to_task, k) for k in candidate_topk_values
    }

    gt_task = class_to_task[targets]
    oracle_shared_pred = method_oracle_shared_local(targets, class_to_task, metrics["shared_local_top1"])
    oracle_expert_pred = method_oracle_expert_local(targets, class_to_task, metrics["expert_local_top1"])

    top2_classes = shared_topk[:, 1]
    top1_class_gap = (
        shared_logits[np.arange(shared_logits.shape[0]), shared_topk[:, 0]]
        - shared_logits[np.arange(shared_logits.shape[0]), top2_classes]
    )
    task_top2 = np.argsort(-block_scores, axis=1)[:, :2]
    top1_task_gap = (
        block_scores[np.arange(block_scores.shape[0]), task_top2[:, 0]]
        - block_scores[np.arange(block_scores.shape[0]), task_top2[:, 1]]
    )
    top1_top2_cross_task = (class_to_task[shared_topk[:, 0]] != class_to_task[shared_topk[:, 1]]).astype(np.float32)

    oracle_summary = {
        "baseline_shared_top1": baseline_acc,
        "oracle_shared_with_gt_task_top1": top1_accuracy(oracle_shared_pred, targets),
        "oracle_expert_with_gt_task_top1": top1_accuracy(oracle_expert_pred, targets),
        "gt_task_in_shared_top2_tasks": float(
            np.mean(
                [
                    gt_task[i] in candidate_tasks_lookup[min(2, max(candidate_topk_values))][i]
                    for i in range(targets.shape[0])
                ]
            )
            * 100.0
        ),
        "gt_task_in_shared_top3_tasks": float(
            np.mean(
                [
                    gt_task[i] in candidate_tasks_lookup[min(3, max(candidate_topk_values))][i]
                    for i in range(targets.shape[0])
                ]
            )
            * 100.0
        ),
        "mean_center_cos_true_task": float(metrics["sim_center_cos"][np.arange(targets.shape[0]), gt_task].mean()),
        "mean_zscore_cos_true_task": float(metrics["sim_zscore_cos"][np.arange(targets.shape[0]), gt_task].mean()),
        "mean_rank_cos_true_task": float(metrics["sim_rank_cos"][np.arange(targets.shape[0]), gt_task].mean()),
        "mean_js_true_task": float(metrics["sim_js"][np.arange(targets.shape[0]), gt_task].mean()),
        "mean_true_minus_best_false_center_cos": float(
            (
                metrics["sim_center_cos"][np.arange(targets.shape[0]), gt_task]
                - np.max(np.where(np.eye(block_scores.shape[1], dtype=bool)[gt_task], -1e9, metrics["sim_center_cos"]), axis=1)
            ).mean()
        ),
        "shared_top1_top2_cross_task_rate": float(top1_top2_cross_task.mean() * 100.0),
    }
    with open(output_dir / "oracle_deep_summary.json", "w", encoding="utf-8") as f:
        json.dump(oracle_summary, f, indent=2)

    grids = grid_values(preset)
    sim_lookup = {
        "center_cos": metrics["sim_center_cos"],
        "zscore_cos": metrics["sim_zscore_cos"],
        "rank_cos": metrics["sim_rank_cos"],
        "top3_masked_cos": metrics["sim_top3_masked_cos"],
        "js": metrics["sim_js"],
    }
    expert_local_lookup = {
        "center": metrics["expert_local_center"],
        "zscore": metrics["expert_local_zscore"],
        "logprob": metrics["expert_local_logprob"],
    }

    rows: List[Dict[str, Any]] = []

    def product_total(*axes: Sequence[Any]) -> int:
        total = 1
        for axis in axes:
            total *= len(axis)
        return total

    def push_row(method_name: str, pred: np.ndarray, **params: Any) -> None:
        row = {
            "method": method_name,
            "top1": round(top1_accuracy(pred, targets), 4),
            "delta_vs_baseline": round(top1_accuracy(pred, targets) - baseline_acc, 4),
            "ambiguous_cross_task_top1": round(
                top1_accuracy(pred[top1_top2_cross_task > 0.5], targets[top1_top2_cross_task > 0.5]), 4
            )
            if np.any(top1_top2_cross_task > 0.5)
            else float("nan"),
        }
        row.update(grouped_correction_stats(pred, baseline_pred, targets))
        row.update(params)
        rows.append(row)

    push_row("baseline_shared", baseline_pred)

    loop_total = product_total(
        grids["sim_variants"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
    )
    for idx, (sim_name, candidate_topk, rerank_topk, alpha, apply_scope, candidate_task_mode) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
            ),
            total=loop_total,
            desc="class_bonus_sim",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "class_bonus_sim")
        pred = method_class_bonus_generic(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            metric=sim_lookup[sim_name],
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha=float(alpha),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )
        push_row(
            "class_bonus_sim",
            pred,
            sim_variant=sim_name,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["gate_quantile"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
    )
    for idx, (sim_name, candidate_topk, rerank_topk, alpha, gate_q, apply_scope, candidate_task_mode) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["gate_quantile"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
            ),
            total=loop_total,
            desc="gated_sim_classgap",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "gated_sim_classgap")
        gate_threshold = float(np.quantile(top1_class_gap, gate_q))
        pred = method_class_bonus_generic(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            metric=sim_lookup[sim_name],
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha=float(alpha),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            gate_mode="class_gap",
            gate_values=top1_class_gap,
            gate_threshold=gate_threshold,
        )
        push_row(
            "gated_class_bonus_sim_classgap",
            pred,
            sim_variant=sim_name,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            gate_quantile=gate_q,
            gate_threshold=gate_threshold,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["gate_quantile"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
    )
    for idx, (sim_name, candidate_topk, rerank_topk, alpha, gate_q, apply_scope, candidate_task_mode) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["gate_quantile"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
            ),
            total=loop_total,
            desc="gated_sim_taskgap",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "gated_sim_taskgap")
        gate_threshold = float(np.quantile(top1_task_gap, gate_q))
        pred = method_class_bonus_generic(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            metric=sim_lookup[sim_name],
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha=float(alpha),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            gate_mode="task_gap",
            gate_values=top1_task_gap,
            gate_threshold=gate_threshold,
        )
        push_row(
            "gated_class_bonus_sim_taskgap",
            pred,
            sim_variant=sim_name,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            gate_quantile=gate_q,
            gate_threshold=gate_threshold,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["ref_mode"],
        grids["sign_mode"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
    )
    for idx, (
        sim_name,
        candidate_topk,
        rerank_topk,
        alpha,
        ref_mode,
        sign_mode,
        apply_scope,
        candidate_task_mode,
    ) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["ref_mode"],
                grids["sign_mode"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
            ),
            total=loop_total,
            desc="relative_sim",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "relative_sim")
        pred = method_class_bonus_generic(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            metric=sim_lookup[sim_name],
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha=float(alpha),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            relative_mode="delta",
            ref_mode=ref_mode,
            sign_mode=sign_mode,
        )
        push_row(
            "relative_class_bonus_sim",
            pred,
            sim_variant=sim_name,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            ref_mode=ref_mode,
            sign_mode=sign_mode,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["ref_mode"],
        grids["sign_mode"],
        grids["gate_quantile"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
    )
    for idx, (
        sim_name,
        candidate_topk,
        rerank_topk,
        alpha,
        ref_mode,
        sign_mode,
        gate_q,
        apply_scope,
        candidate_task_mode,
    ) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["ref_mode"],
                grids["sign_mode"],
                grids["gate_quantile"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
            ),
            total=loop_total,
            desc="gated_relative_sim",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "gated_relative_sim")
        gate_threshold = float(np.quantile(top1_task_gap, gate_q))
        pred = method_class_bonus_generic(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            metric=sim_lookup[sim_name],
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha=float(alpha),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            relative_mode="delta",
            ref_mode=ref_mode,
            sign_mode=sign_mode,
            gate_mode="task_gap",
            gate_values=top1_task_gap,
            gate_threshold=gate_threshold,
        )
        push_row(
            "gated_relative_class_bonus_sim_taskgap",
            pred,
            sim_variant=sim_name,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            ref_mode=ref_mode,
            sign_mode=sign_mode,
            gate_quantile=gate_q,
            gate_threshold=gate_threshold,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        grids["local_variant"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["beta_local"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
        grids["multiply_mode"],
    )
    for idx, (
        sim_name,
        local_variant,
        candidate_topk,
        rerank_topk,
        alpha,
        beta_local,
        apply_scope,
        candidate_task_mode,
        multiply_mode,
    ) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                grids["local_variant"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["beta_local"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
                grids["multiply_mode"],
            ),
            total=loop_total,
            desc="expert_local",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "expert_local")
        pred = method_class_bonus_expert_local(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            task_metric=sim_lookup[sim_name],
            expert_local_metric=expert_local_lookup[local_variant],
            class_to_task=class_to_task,
            task_starts=task_starts,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha_task=float(alpha),
            beta_local=float(beta_local),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            multiply_mode=multiply_mode,
        )
        push_row(
            "class_bonus_sim_expert_local",
            pred,
            sim_variant=sim_name,
            local_variant=local_variant,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            beta_local=beta_local,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            multiply_mode=multiply_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        grids["local_variant"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["beta_local"],
        grids["gate_quantile"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
        grids["multiply_mode"],
    )
    for idx, (
        sim_name,
        local_variant,
        candidate_topk,
        rerank_topk,
        alpha,
        beta_local,
        gate_q,
        apply_scope,
        candidate_task_mode,
        multiply_mode,
    ) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                grids["local_variant"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["beta_local"],
                grids["gate_quantile"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
                grids["multiply_mode"],
            ),
            total=loop_total,
            desc="gated_expert_local",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "gated_expert_local")
        gate_threshold = float(np.quantile(top1_class_gap, gate_q))
        pred = method_class_bonus_expert_local(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            task_metric=sim_lookup[sim_name],
            expert_local_metric=expert_local_lookup[local_variant],
            class_to_task=class_to_task,
            task_starts=task_starts,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha_task=float(alpha),
            beta_local=float(beta_local),
            rerank_topk=int(rerank_topk),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            gate_mode="class_gap",
            gate_values=top1_class_gap,
            gate_threshold=gate_threshold,
            multiply_mode=multiply_mode,
        )
        push_row(
            "gated_class_bonus_sim_expert_local_classgap",
            pred,
            sim_variant=sim_name,
            local_variant=local_variant,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha=alpha,
            beta_local=beta_local,
            gate_quantile=gate_q,
            gate_threshold=gate_threshold,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
            multiply_mode=multiply_mode,
        )

    loop_total = product_total(
        grids["sim_variants"],
        candidate_topk_values,
        rerank_topk_values,
        grids["alpha"],
        grids["kappa"],
        grids["gate_quantile"],
        grids["apply_scope"],
        grids["candidate_task_mode"],
    )
    for idx, (sim_name, candidate_topk, rerank_topk, alpha, kappa, gate_q, apply_scope, candidate_task_mode) in enumerate(
        maybe_tqdm(
            itertools.product(
                grids["sim_variants"],
                candidate_topk_values,
                rerank_topk_values,
                grids["alpha"],
                grids["kappa"],
                grids["gate_quantile"],
                grids["apply_scope"],
                grids["candidate_task_mode"],
            ),
            total=loop_total,
            desc="adaptive_alpha",
            enabled=show_progress,
        ),
        start=1,
    ):
        progress_log(show_progress, idx, loop_total, "adaptive_alpha")
        tau = float(np.quantile(top1_task_gap, gate_q))
        pred = method_class_bonus_adaptive_alpha(
            shared_logits=shared_logits,
            shared_topk=shared_topk,
            block_scores=block_scores,
            metric=sim_lookup[sim_name],
            class_to_task=class_to_task,
            candidate_tasks_by_k=candidate_tasks_lookup[candidate_topk],
            alpha0=float(alpha),
            rerank_topk=int(rerank_topk),
            uncertainty=top1_task_gap,
            tau=tau,
            kappa=float(kappa),
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )
        push_row(
            "adaptive_class_bonus_sim_taskgap",
            pred,
            sim_variant=sim_name,
            candidate_topk=candidate_topk,
            rerank_topk=rerank_topk,
            alpha0=alpha,
            kappa=kappa,
            tau_quantile=gate_q,
            tau=tau,
            apply_scope=apply_scope,
            candidate_task_mode=candidate_task_mode,
        )

    rows_sorted = sorted(rows, key=lambda x: (-x["top1"], x["method"]))
    all_csv = output_dir / "fusion_sweep_cos_deep_all.csv"
    best_csv = output_dir / "fusion_sweep_cos_deep_best_per_method.csv"
    all_fieldnames = sorted({key for row in rows_sorted for key in row.keys()})

    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row)

    best_rows = []
    seen_methods = set()
    for row in rows_sorted:
        if row["method"] in seen_methods:
            continue
        seen_methods.add(row["method"])
        best_rows.append(row)

    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow(row)

    print("\n[summary] Oracle analysis")
    for k, v in oracle_summary.items():
        print(f"  {k}: {v}")

    print("\n[summary] Best per method")
    for row in best_rows:
        print(f"  {row['method']:<40} top1={row['top1']:.4f} delta={row['delta_vs_baseline']:+.4f}")

    print(f"\n[done] Saved full sweep to: {all_csv}")
    print(f"[done] Saved best-per-method to: {best_csv}")
    print(f"[done] Saved oracle summary to: {output_dir / 'oracle_deep_summary.json'}")


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    norm_args = normalize_args(cfg, args.device, args.batch_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if norm_args.get("model_name", "spie_v16").lower() != "spie_v16":
        print(f"[warn] model_name in config is {norm_args.get('model_name')!r}; overriding to 'spie_v16'.")
        norm_args["model_name"] = "spie_v16"

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
    rebuild_spie_v16_to_checkpoint(learner, data_manager, checkpoint)
    learner._network.to(norm_args["device"][0])
    learner._network.eval()

    loader = build_test_loader(
        data_manager=data_manager,
        total_classes=learner._total_classes,
        batch_size=norm_args["batch_size"],
        num_workers=args.num_workers,
    )

    output_dir, cache_file = cache_paths(args.output_dir, args.cache_name)
    cache = extract_or_load_logits(
        learner,
        loader,
        args.force_recache,
        cache_file,
        args.base_cache,
        args.progress,
    )

    class_to_task = build_class_to_task(cache["task_starts"], cache["task_ends"])
    metrics = precompute_metrics(
        shared_logits=cache["shared_logits"],
        expert_logits=cache["expert_logits"],
        task_starts=cache["task_starts"],
        task_ends=cache["task_ends"],
        local_topk=int(norm_args.get("verifier_local_topk", 3)),
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
    print("  base_cache:", args.base_cache)

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
        max_shared_topk=args.max_shared_topk,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
