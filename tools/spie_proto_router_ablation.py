#!/usr/bin/env python3
"""
Eval-only prototype-router ablation for SPiE.

This script is intentionally a companion of tools/spie_ablation.py:
- it loads an already-trained SPiE checkpoint,
- builds the incremental model structure without training,
- collects shared logits / shared features / expert logits,
- evaluates several task routers without retraining anything.

Recommended use:
python tools/spie_proto_activation_router_ablation.py \
  --config exps/spie/xxx.json \
  --checkpoint logs/.../checkpoints/task_9.pkl \
  --device 0 \
  --output-json logs/.../proto_router_task9.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

# Allow running this script directly from tools/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import factory
from utils.data_manager import DataManager


# -------------------------
# Basic utils
# -------------------------

def set_random(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)


def normalize_np(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), eps, None)


def parse_csv_floats(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if np.isnan(obj):
            return "nan"
        return obj
    return obj


def print_metric(name, accy, task_acc=None):
    print(f"\n[{name}]")
    print(f"top1: {accy['top1']:.2f}")
    print(f"top5: {accy['top5']:.2f}")
    print("grouped:", accy["grouped"])
    if task_acc is not None:
        print(f"task_acc: {task_acc:.2f}")


def parse_device(device_arg: str):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def get_task_ranges(model):
    return list(model.task_class_ranges)


def class_to_task_ids(model, y):
    y = np.asarray(y)
    task_ids = np.full_like(y, fill_value=-1, dtype=np.int64)
    for task_id, (start, end) in enumerate(get_task_ranges(model)):
        mask = (y >= start) & (y < end)
        task_ids[mask] = task_id
    if np.any(task_ids < 0):
        bad = y[task_ids < 0][:10]
        raise ValueError(f"Some labels/classes are outside task ranges: {bad}")
    return task_ids


# -------------------------
# Model construction
# -------------------------

def build_model_structure_without_training(model, data_manager, num_tasks):
    """
    Run incremental_train only to initialize task structure, heads, and loaders.
    model._train is temporarily replaced with a no-op so no training is run.
    """
    original_train = model._train

    def noop_train(train_loader):
        return None

    model._train = noop_train
    try:
        for _ in range(num_tasks):
            model.incremental_train(data_manager)
            backbone = (
                model._backbone_module()
                if hasattr(model, "_backbone_module")
                else model._network.backbone
            )
            if hasattr(backbone, "adapter_update"):
                backbone.adapter_update()
            model.after_task()
    finally:
        model._train = original_train


def set_eval_known_classes(model, ckpt):
    """
    Checkpoints are saved after after_task(), so known_classes often equals total_classes.
    For grouped old/new accuracy, use the current task start.
    """
    ckpt_known_classes = int(ckpt["known_classes"])
    total_classes = int(ckpt["total_classes"])
    if ckpt_known_classes >= total_classes and model.task_class_ranges:
        model._known_classes = int(model.task_class_ranges[-1][0])
    else:
        model._known_classes = ckpt_known_classes


# -------------------------
# Collect logits / features
# -------------------------

@torch.no_grad()
def collect_eval_pack_np(model, loader):
    """
    Returns:
      shared_logits_np: [N, C]
      shared_features_np: [N, D]
      expert_logits_by_task: list of [N, C_t]
      y_true: [N]
    """
    model._network.eval()

    all_shared_logits = []
    all_shared_features = []
    all_targets = []

    num_tasks = len(model.task_class_ranges)
    expert_logits_chunks = [[] for _ in range(num_tasks)]

    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(model._device)

        # Use the shared branch once and keep its feature.
        res = model._network.backbone(inputs, adapter_id=-1, train=False)
        shared_features = res["cls_features"]
        shared_logits = model._network.fc_shared_cls(shared_features)["logits"][
            :, : model._total_classes
        ]

        expert_logits_map = model._collect_expert_logits(
            inputs, list(range(num_tasks))
        ) if num_tasks > 0 else {}

        all_shared_features.append(shared_features.cpu().numpy().astype(np.float32))
        all_shared_logits.append(shared_logits.cpu().numpy().astype(np.float32))
        all_targets.append(targets.numpy())

        for task_id in range(num_tasks):
            expert_logits_chunks[task_id].append(
                expert_logits_map[task_id].cpu().numpy().astype(np.float32)
            )

    shared_logits_np = np.concatenate(all_shared_logits, axis=0)
    shared_features_np = np.concatenate(all_shared_features, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    expert_logits_by_task = [
        np.concatenate(task_chunks, axis=0)
        if task_chunks
        else np.zeros((shared_logits_np.shape[0], 0), dtype=np.float32)
        for task_chunks in expert_logits_chunks
    ]

    return shared_logits_np, shared_features_np, expert_logits_by_task, y_true


# -------------------------
# Prototype extraction
# -------------------------

def _find_last_linear_weight(module: nn.Module):
    """
    Robustly fetch the actual Linear weight inside TunaLinear heads.
    Works for Sequential(Linear) and Sequential(LayerNorm, Linear).
    """
    last_weight = None
    for submodule in module.modules():
        if hasattr(submodule, "weight"):
            weight = getattr(submodule, "weight")
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                last_weight = weight
    return last_weight


def get_shared_weight_prototypes_np(model):
    """
    Classifier-induced prototypes from fc_shared_cls weights.
    For TunaLinear, fc_shared_cls.heads stores one head per task.
    """
    fc = model._network.fc_shared_cls

    if hasattr(fc, "heads"):
        weights = []
        for head in fc.heads:
            w = _find_last_linear_weight(head)
            if w is None:
                raise RuntimeError(f"Cannot find Linear weight in shared head: {head}")
            weights.append(w.detach().cpu())
        weight = torch.cat(weights, dim=0)
    elif hasattr(fc, "weight"):
        weight = fc.weight.detach().cpu()
    else:
        raise RuntimeError(f"Unsupported fc_shared_cls type: {type(fc)}")

    weight_np = weight[: model._total_classes].numpy().astype(np.float32)
    return normalize_np(weight_np, axis=1)


def build_mean_prototypes_np(model, loader, weight_proto_fallback=None):
    """
    Feature-mean class prototypes from the frozen shared branch.

    The calibration loader should cover seen train classes [0, total_classes).
    No gradient and no retraining are used.
    """
    model._network.eval()

    total_classes = int(model._total_classes)
    sums = None
    counts = np.zeros((total_classes,), dtype=np.int64)

    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(model._device)
            targets_np = targets.numpy()

            res = model._network.backbone(inputs, adapter_id=-1, train=False)
            feat = res["cls_features"]
            feat_np = F.normalize(feat, p=2, dim=1).cpu().numpy().astype(np.float32)

            if sums is None:
                sums = np.zeros((total_classes, feat_np.shape[1]), dtype=np.float32)

            for local_i, cls in enumerate(targets_np.tolist()):
                if 0 <= int(cls) < total_classes:
                    sums[int(cls)] += feat_np[local_i]
                    counts[int(cls)] += 1

    if sums is None:
        raise RuntimeError("Empty loader while building mean prototypes.")

    mean_proto = sums.copy()
    missing = counts == 0
    if np.any(missing):
        if weight_proto_fallback is None:
            raise RuntimeError(
                f"Missing classes in prototype loader: {np.where(missing)[0][:20]}"
            )
        mean_proto[missing] = weight_proto_fallback[missing]

    return normalize_np(mean_proto, axis=1), counts


def make_hybrid_prototypes_np(weight_proto, mean_proto, eta: float):
    """
    eta=1.0 -> pure classifier-weight prototype.
    eta=0.0 -> pure feature-mean prototype.
    """
    proto = eta * weight_proto + (1.0 - eta) * mean_proto
    return normalize_np(proto.astype(np.float32), axis=1)



def row_zscore_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Standardize each sample's class-energy vector before mixing evidence types."""
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return ((x - mean) / np.clip(std, eps, None)).astype(np.float32)


def build_mean_var_prototypes_np(
    model,
    loader,
    weight_proto_fallback=None,
    var_floor: float = 1e-4,
    shrinkage: float = 0.0,
):
    """
    One-pass feature statistics from the frozen shared branch.

    Returns:
      mean_proto_cos: L2-normalized class mean, for cosine prototype routing.
      counts: number of feature samples per class.
      mean_gauss: non-normalized mean of normalized features, for Gaussian energy.
      var_gauss: diagonal variance of normalized features, for Gaussian energy.

    This is eval-only: no gradient, no classifier update, no retraining.
    """
    model._network.eval()

    total_classes = int(model._total_classes)
    sums = None
    sq_sums = None
    counts = np.zeros((total_classes,), dtype=np.int64)

    with torch.no_grad():
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(model._device)
            targets_np = targets.numpy()

            res = model._network.backbone(inputs, adapter_id=-1, train=False)
            feat = res["cls_features"]
            feat_np = F.normalize(feat, p=2, dim=1).cpu().numpy().astype(np.float32)

            if sums is None:
                dim = feat_np.shape[1]
                sums = np.zeros((total_classes, dim), dtype=np.float64)
                sq_sums = np.zeros((total_classes, dim), dtype=np.float64)

            for local_i, cls in enumerate(targets_np.tolist()):
                cls = int(cls)
                if 0 <= cls < total_classes:
                    z = feat_np[local_i].astype(np.float64)
                    sums[cls] += z
                    sq_sums[cls] += z * z
                    counts[cls] += 1

    if sums is None:
        raise RuntimeError("Empty loader while building mean/variance prototypes.")

    total_count = max(int(counts.sum()), 1)
    global_mean = sums.sum(axis=0) / total_count
    global_var = sq_sums.sum(axis=0) / total_count - global_mean * global_mean
    global_var = np.clip(global_var, var_floor, None)

    safe_counts = np.maximum(counts[:, None], 1)
    mean_gauss = sums / safe_counts
    var_gauss = sq_sums / safe_counts - mean_gauss * mean_gauss

    missing = counts == 0
    if np.any(missing):
        if weight_proto_fallback is None:
            raise RuntimeError(
                f"Missing classes in prototype loader: {np.where(missing)[0][:20]}"
            )
        mean_gauss[missing] = weight_proto_fallback[missing]
        var_gauss[missing] = global_var[None, :]

    shrinkage = min(max(float(shrinkage), 0.0), 1.0)
    if shrinkage > 0.0:
        var_gauss = (1.0 - shrinkage) * var_gauss + shrinkage * global_var[None, :]

    var_gauss = np.clip(var_gauss, var_floor, None)
    mean_gauss = mean_gauss.astype(np.float32)
    var_gauss = var_gauss.astype(np.float32)
    mean_proto_cos = normalize_np(mean_gauss, axis=1)

    return mean_proto_cos, counts, mean_gauss, var_gauss


def gaussian_class_energy_diag_np(
    shared_features,
    mean_gauss,
    var_gauss,
    gaussian_scale: float = 1.0,
    use_logdet: bool = True,
    var_floor: float = 1e-4,
):
    """
    Diagonal-Gaussian class energy from stored feature mean/variance.

    Uses the dimension-averaged Gaussian negative distance, not the full summed
    log-likelihood, so the magnitude is stable across feature dimensions:

      E_c(z) = -0.5 * mean_j [ (z_j - mu_cj)^2 / var_cj + log(var_cj) ]

    If use_logdet=False, the log-variance term is omitted and this becomes a
    variance-aware Mahalanobis prototype router.
    """
    z = normalize_np(shared_features.astype(np.float32), axis=1)
    mu = mean_gauss.astype(np.float32)
    var = np.clip(var_gauss.astype(np.float32), float(var_floor), None)
    inv_var = 1.0 / var
    dim = float(z.shape[1])

    # Efficient expansion of sum_j (z_j - mu_cj)^2 / var_cj.
    z2 = z * z
    mu_inv = mu * inv_var
    mu2_inv_sum = np.sum(mu * mu * inv_var, axis=1, keepdims=True).T
    maha = z2 @ inv_var.T - 2.0 * (z @ mu_inv.T) + mu2_inv_sum
    maha_mean = maha / max(dim, 1.0)

    if use_logdet:
        logdet_mean = np.mean(np.log(var), axis=1, keepdims=True).T
        energy = -0.5 * (maha_mean + logdet_mean)
    else:
        energy = -0.5 * maha_mean

    return (float(gaussian_scale) * energy).astype(np.float32)


# -------------------------
# Router / fusion
# -------------------------

def class_energy_from_prototypes_np(shared_features, prototypes, proto_scale=1.0):
    z = normalize_np(shared_features.astype(np.float32), axis=1)
    p = normalize_np(prototypes.astype(np.float32), axis=1)
    return (float(proto_scale) * (z @ p.T)).astype(np.float32)


def task_posterior_from_class_energy_np(
    model,
    class_energy,
    task_temperature=1.0,
    topk=None,
):
    """
    Prototype-set affinity:
      A_t(x) = logmeanexp_{c in C_t} energy_c(x)
      q(t|x) = softmax_t A_t(x)

    If topk is not None, only the top-k prototype energies within each task
    are aggregated.
    """
    temp = max(float(task_temperature), 1e-6)
    num_samples = class_energy.shape[0]
    task_ranges = get_task_ranges(model)
    task_logits = np.zeros((num_samples, len(task_ranges)), dtype=np.float32)

    for task_id, (start, end) in enumerate(task_ranges):
        block = class_energy[:, start:end] / temp
        if block.shape[1] == 0:
            task_logits[:, task_id] = -np.inf
            continue

        if topk is not None:
            k = max(1, min(int(topk), block.shape[1]))
            # Unsorted top-k is enough for logmeanexp.
            block = np.partition(block, kth=block.shape[1] - k, axis=1)[:, -k:]

        max_block = np.max(block, axis=1, keepdims=True)
        # logmeanexp, not logsumexp, so task score is less biased by task width.
        task_logits[:, task_id] = np.squeeze(
            max_block
            + np.log(np.mean(np.exp(block - max_block), axis=1, keepdims=True) + 1e-12),
            axis=1,
        )

    return stable_softmax_np(task_logits).astype(np.float32), task_logits



def task_posterior_from_global_argmax_np(model, class_energy):
    """
    Hard class-first router.

    First choose the globally most activated class prototype:
        c*(x) = argmax_c E_c(x)
    Then route to the task containing that class:
        q(t|x) = 1[t = task(c*)]

    This directly tests whether the class-level CRFC/prototype prediction
    already gives a good task id. It has no temperature and no learned weight.
    """
    top_class = np.argmax(class_energy, axis=1).astype(np.int64)
    top_task = class_to_task_ids(model, top_class)
    p_task = np.zeros((class_energy.shape[0], len(get_task_ranges(model))), dtype=np.float32)
    p_task[np.arange(class_energy.shape[0]), top_task] = 1.0
    return p_task


def task_posterior_from_task_max_np(
    model,
    class_energy,
    task_temperature=1.0,
):
    """
    Soft prototype-activation router.

    Each task is represented by its most activated class prototype:
        A_t(x) = max_{c in C_t} E_c(x)
        q(t|x) = softmax_t(A_t(x) / T)

    Compared with full log-mean-exp, this preserves the class classifier's
    max-score decision principle while still returning a soft posterior.
    """
    temp = max(float(task_temperature), 1e-6)
    num_samples = class_energy.shape[0]
    task_ranges = get_task_ranges(model)
    task_logits = np.zeros((num_samples, len(task_ranges)), dtype=np.float32)

    for task_id, (start, end) in enumerate(task_ranges):
        block = class_energy[:, start:end] / temp
        if block.shape[1] == 0:
            task_logits[:, task_id] = -np.inf
        else:
            task_logits[:, task_id] = np.max(block, axis=1)

    return stable_softmax_np(task_logits).astype(np.float32), task_logits


def task_posterior_from_global_topk_energy_np(
    model,
    class_energy,
    global_topk: int,
    task_temperature=1.0,
):
    """
    Global top-k activation router.

    Only the globally top-k activated class prototypes are allowed to vote.
    For each task, aggregate the energies of its selected prototypes:
        S_k(x) = TopK_c E_c(x)
        A_t(x) = LogSumExp_{c in S_k(x) cap C_t} E_c(x)
        q(t|x) = softmax_t(A_t(x) / T)

    This avoids a failure mode of full task log-mean-exp: irrelevant medium-score
    classes inside a wrong task cannot collectively lift that task unless they
    are among the global top-k competitors.
    """
    k = max(1, min(int(global_topk), class_energy.shape[1]))
    temp = max(float(task_temperature), 1e-6)
    num_samples = class_energy.shape[0]
    task_ranges = get_task_ranges(model)
    task_logits = np.full((num_samples, len(task_ranges)), -np.inf, dtype=np.float32)

    # Top-k largest energies for each sample.
    top_idx = np.argpartition(class_energy, kth=class_energy.shape[1] - k, axis=1)[:, -k:]
    top_vals = np.take_along_axis(class_energy, top_idx, axis=1) / temp

    for task_id, (start, end) in enumerate(task_ranges):
        in_task = (top_idx >= start) & (top_idx < end)
        vals = np.where(in_task, top_vals, -np.inf)
        max_vals = np.max(vals, axis=1, keepdims=True)
        has_any = np.isfinite(max_vals[:, 0])
        if not np.any(has_any):
            continue

        safe_vals = vals[has_any]
        safe_max = max_vals[has_any]
        task_logits[has_any, task_id] = np.squeeze(
            safe_max + np.log(np.sum(np.exp(safe_vals - safe_max), axis=1, keepdims=True) + 1e-12),
            axis=1,
        )

    return stable_softmax_np(task_logits).astype(np.float32), task_logits

def task_posterior_oracle_np(model, y_true):
    task_ids = class_to_task_ids(model, y_true)
    p_task = np.zeros((len(y_true), len(get_task_ranges(model))), dtype=np.float32)
    p_task[np.arange(len(y_true)), task_ids] = 1.0
    return p_task


def posterior_fusion_with_task_posterior_np(
    model,
    shared_logits,
    expert_logits_by_task,
    p_task,
):
    shared_temperature = max(float(model.posterior_shared_temperature), 1e-6)
    expert_temperature = max(float(model.posterior_expert_temperature), 1e-6)

    p_shared = stable_softmax_np(shared_logits / shared_temperature).astype(np.float32)
    p_moe = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start, end) in enumerate(get_task_ranges(model)):
        expert_logits = expert_logits_by_task[task_id]
        if expert_logits.shape[1] == 0:
            continue
        local_prob = stable_softmax_np(expert_logits / expert_temperature).astype(np.float32)
        p_moe[:, start:end] = p_task[:, task_id : task_id + 1] * local_prob

    alpha = min(max(float(model.posterior_alpha), 0.0), 1.0)
    p_final = alpha * p_shared + (1.0 - alpha) * p_moe
    return p_moe, p_final


def evaluate_router(
    model,
    name,
    p_task,
    shared_logits_np,
    expert_logits_by_task,
    y_true,
):
    p_moe, p_final = posterior_fusion_with_task_posterior_np(
        model, shared_logits_np, expert_logits_by_task, p_task
    )

    p_moe_pred = model._predict_topk_np(p_moe)
    p_final_pred = model._predict_topk_np(p_final)

    p_moe_accy = model._evaluate(p_moe_pred, y_true)
    p_final_accy = model._evaluate(p_final_pred, y_true)

    true_task = class_to_task_ids(model, y_true)
    task_acc = 100.0 * float((np.argmax(p_task, axis=1) == true_task).mean())

    print_metric(f"{name}_p_moe", p_moe_accy, task_acc=task_acc)
    print_metric(f"{name}_p_final", p_final_accy, task_acc=task_acc)

    return {
        "task_acc": task_acc,
        "p_moe": p_moe_accy,
        "p_final": p_final_accy,
    }


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SPiE eval-only prototype-router ablation."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the SPiE experiment json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved task_*.pkl checkpoint.")
    parser.add_argument("--device", type=str, default="0", help="CUDA id, or 'cpu'. Default: 0.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Prototype/calibration loader batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Prototype/calibration loader worker count.")
    parser.add_argument("--proto-scale", type=float, default=1.0, help="Scale for prototype cosine energy. Use 1.0 to match TunaLinear-style cosine logits.")
    parser.add_argument("--hybrid-etas", type=str, default="0.25,0.5,0.75", help="Comma-separated eta values: eta*weight + (1-eta)*mean.")
    parser.add_argument("--topk-values", type=str, default="1,3,5", help="Comma-separated top-k values for top-k prototype routing.")
    parser.add_argument("--activation-topk-values", type=str, default="1,3,5", help="Comma-separated global top-k values for activation-based task routers.")
    parser.add_argument("--gaussian-scale", type=float, default=1.0, help="Scale for diagonal-Gaussian prototype energy.")
    parser.add_argument("--gaussian-var-floor", type=float, default=1e-4, help="Variance floor for diagonal-Gaussian prototype energy.")
    parser.add_argument("--gaussian-shrinkage", type=float, default=0.1, help="Shrink class variances toward global variance. 0 disables shrinkage.")
    parser.add_argument("--no-gaussian-logdet", action="store_true", help="Omit log-variance term; use pure diagonal Mahalanobis energy.")
    parser.add_argument("--energy-mix-lambdas", type=str, default="0.25,0.5,0.75,0.9", help="Comma-separated lambda values for lambda*weight_energy + (1-lambda)*gaussian_energy.")
    parser.add_argument("--output-json", "--output", dest="output_json", type=str, default=None, help="Optional path to save summary json.")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        args = json.load(f)

    if args.get("model_name", "").lower() != "spie":
        raise ValueError(
            f"This script only supports model_name='spie', got {args.get('model_name')!r}."
        )

    args["swanlab"] = False
    args["spie_backbone_dataparallel"] = False

    if isinstance(args.get("seed"), list):
        args["seed"] = args["seed"][0]
    if args_cli.seed is not None:
        args["seed"] = args_cli.seed

    device = parse_device(args_cli.device)
    args["device"] = [device]
    set_random(int(args["seed"]))

    ckpt = torch.load(args_cli.checkpoint, map_location="cpu")
    target_task = int(ckpt["tasks"])
    num_tasks = target_task + 1

    print(f"Loaded checkpoint: {args_cli.checkpoint}")
    print(f"Checkpoint task: {target_task}")
    print(f"Checkpoint total_classes: {ckpt['total_classes']}")

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    if num_tasks > data_manager.nb_tasks:
        raise ValueError(
            f"Checkpoint asks for {num_tasks} tasks, but config/data define only {data_manager.nb_tasks} tasks."
        )

    model = factory.get_model(args["model_name"], args)
    build_model_structure_without_training(model, data_manager, num_tasks)

    model._cur_task = target_task
    model._total_classes = int(ckpt["total_classes"])
    set_eval_known_classes(model, ckpt)

    missing, unexpected = model._network.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )
    if missing:
        print("\n[Warning] Missing keys:")
        for key in missing[:20]:
            print(" ", key)
        if len(missing) > 20:
            print(f" ... {len(missing) - 20} more")
    if unexpected:
        print("\n[Warning] Unexpected keys:")
        for key in unexpected[:20]:
            print(" ", key)
        if len(unexpected) > 20:
            print(f" ... {len(unexpected) - 20} more")

    model._network.to(device)
    model._network.eval()

    # -------------------------
    # Collect test logits/features
    # -------------------------
    print("\nCollecting test logits/features...")
    shared_logits_np, shared_features_np, expert_logits_by_task, y_true = collect_eval_pack_np(
        model, model.test_loader
    )

    # -------------------------
    # Original baselines
    # -------------------------
    shared_pred = model._predict_topk_np(shared_logits_np)
    shared_accy = model._evaluate(shared_pred, y_true)

    original_p_moe, original_p_final = model._posterior_fusion_probs_np(
        shared_logits_np,
        expert_logits_by_task,
    )
    original_p_moe_accy = model._evaluate(model._predict_topk_np(original_p_moe), y_true)
    original_p_final_accy = model._evaluate(model._predict_topk_np(original_p_final), y_true)

    # Re-derive original task posterior as logmeanexp over shared logits.
    p_task_shared_logit, _ = task_posterior_from_class_energy_np(
        model,
        shared_logits_np,
        task_temperature=float(model.posterior_task_temperature),
        topk=None,
    )
    shared_logit_router_result = evaluate_router(
        model,
        "shared_logit_router_reimpl",
        p_task_shared_logit,
        shared_logits_np,
        expert_logits_by_task,
        y_true,
    )

    print_metric("shared_fc", shared_accy)
    print_metric("original_p_moe", original_p_moe_accy)
    print_metric("original_p_final", original_p_final_accy)

    # -------------------------
    # Build explicit prototypes
    # -------------------------
    print("\nBuilding explicit prototypes from frozen shared branch...")

    batch_size = (
        args_cli.batch_size
        if args_cli.batch_size is not None
        else int(args.get("batch_size", 128))
    )
    num_workers = (
        args_cli.num_workers
        if args_cli.num_workers is not None
        else int(args.get("num_workers", 8))
    )

    proto_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="train",
        mode="test",
    )
    proto_loader = DataLoader(
        proto_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    weight_proto = get_shared_weight_prototypes_np(model)
    mean_proto, mean_counts, mean_gauss, var_gauss = build_mean_var_prototypes_np(
        model,
        proto_loader,
        weight_proto_fallback=weight_proto,
        var_floor=float(args_cli.gaussian_var_floor),
        shrinkage=float(args_cli.gaussian_shrinkage),
    )

    print(
        "Prototype counts:",
        f"min={int(mean_counts.min())}",
        f"max={int(mean_counts.max())}",
        f"missing={int((mean_counts == 0).sum())}",
    )
    print(
        "Gaussian variance:",
        f"floor={float(args_cli.gaussian_var_floor):.2e}",
        f"shrinkage={float(args_cli.gaussian_shrinkage):.2f}",
        f"min={float(var_gauss.min()):.2e}",
        f"max={float(var_gauss.max()):.2e}",
    )

    # -------------------------
    # Prototype-router variants
    # -------------------------
    results = {
        "shared_fc": shared_accy,
        "original_p_moe": original_p_moe_accy,
        "original_p_final": original_p_final_accy,
        "shared_logit_router_reimpl": shared_logit_router_result,
    }

    def run_proto_router(name, proto, topk=None):
        class_energy = class_energy_from_prototypes_np(
            shared_features_np,
            proto,
            proto_scale=float(args_cli.proto_scale),
        )
        p_task, task_logits = task_posterior_from_class_energy_np(
            model,
            class_energy,
            task_temperature=float(model.posterior_task_temperature),
            topk=topk,
        )
        result = evaluate_router(
            model,
            name,
            p_task,
            shared_logits_np,
            expert_logits_by_task,
            y_true,
        )
        result["task_logits_summary"] = {
            "min": float(np.min(task_logits)),
            "max": float(np.max(task_logits)),
            "mean": float(np.mean(task_logits)),
        }
        return result

    def run_energy_router(name, class_energy, topk=None):
        p_task, task_logits = task_posterior_from_class_energy_np(
            model,
            class_energy,
            task_temperature=float(model.posterior_task_temperature),
            topk=topk,
        )
        result = evaluate_router(
            model,
            name,
            p_task,
            shared_logits_np,
            expert_logits_by_task,
            y_true,
        )
        result["task_logits_summary"] = {
            "min": float(np.min(task_logits)),
            "max": float(np.max(task_logits)),
            "mean": float(np.mean(task_logits)),
        }
        return result

    results["weight_proto_router"] = run_proto_router(
        "weight_proto_router",
        weight_proto,
        topk=None,
    )

    results["mean_proto_router"] = run_proto_router(
        "mean_proto_router",
        mean_proto,
        topk=None,
    )

    gaussian_energy = gaussian_class_energy_diag_np(
        shared_features_np,
        mean_gauss,
        var_gauss,
        gaussian_scale=float(args_cli.gaussian_scale),
        use_logdet=not bool(args_cli.no_gaussian_logdet),
        var_floor=float(args_cli.gaussian_var_floor),
    )
    gaussian_name = "diag_gaussian_router" if not args_cli.no_gaussian_logdet else "diag_mahalanobis_router"
    results[gaussian_name] = run_energy_router(
        gaussian_name,
        gaussian_energy,
        topk=None,
    )

    # Energy-level hybrid: discriminative classifier-weight energy + generative Gaussian energy.
    # Per-sample z-scoring keeps the two evidence types on comparable scales.
    weight_energy_for_mix = class_energy_from_prototypes_np(
        shared_features_np,
        weight_proto,
        proto_scale=float(args_cli.proto_scale),
    )

    # -------------------------
    # Activation-based routers
    # -------------------------
    # These are the least heuristic checks:
    #   1) global argmax class -> task id
    #   2) per-task max prototype activation
    #   3) global top-k activated prototypes vote for tasks
    #
    # They answer the question:
    #   If the CRFC / classifier-induced prototype has high class accuracy,
    #   can its class activations be converted into task routing without
    #   averaging over many irrelevant classes?
    p_task_argmax = task_posterior_from_global_argmax_np(model, weight_energy_for_mix)
    results["weight_proto_argmax_task_router"] = evaluate_router(
        model,
        "weight_proto_argmax_task_router",
        p_task_argmax,
        shared_logits_np,
        expert_logits_by_task,
        y_true,
    )

    p_task_max, task_logits_max = task_posterior_from_task_max_np(
        model,
        weight_energy_for_mix,
        task_temperature=float(model.posterior_task_temperature),
    )
    results["weight_proto_task_max_router"] = evaluate_router(
        model,
        "weight_proto_task_max_router",
        p_task_max,
        shared_logits_np,
        expert_logits_by_task,
        y_true,
    )
    results["weight_proto_task_max_router"]["task_logits_summary"] = {
        "min": float(np.min(task_logits_max)),
        "max": float(np.max(task_logits_max)),
        "mean": float(np.mean(task_logits_max)),
    }

    activation_topk_values = parse_csv_ints(args_cli.activation_topk_values)
    for k in activation_topk_values:
        p_task_topk, task_logits_topk = task_posterior_from_global_topk_energy_np(
            model,
            weight_energy_for_mix,
            global_topk=k,
            task_temperature=float(model.posterior_task_temperature),
        )
        key = f"global_top{k}_weight_proto_router"
        results[key] = evaluate_router(
            model,
            key,
            p_task_topk,
            shared_logits_np,
            expert_logits_by_task,
            y_true,
        )
        results[key]["task_logits_summary"] = {
            "min": float(np.min(task_logits_topk)),
            "max": float(np.max(task_logits_topk)),
            "mean": float(np.mean(task_logits_topk)),
        }

    weight_energy_z = row_zscore_np(weight_energy_for_mix)
    gaussian_energy_z = row_zscore_np(gaussian_energy)

    mix_lambdas = parse_csv_floats(args_cli.energy_mix_lambdas)
    for lam in mix_lambdas:
        lam = min(max(float(lam), 0.0), 1.0)
        mixed_energy = lam * weight_energy_z + (1.0 - lam) * gaussian_energy_z
        key = f"energy_mix_router_lambda{lam:g}"
        results[key] = run_energy_router(key, mixed_energy, topk=None)

    etas = parse_csv_floats(args_cli.hybrid_etas)
    topk_values = parse_csv_ints(args_cli.topk_values)

    hybrid_cache = {}
    for eta in etas:
        proto = make_hybrid_prototypes_np(weight_proto, mean_proto, eta=eta)
        key = f"hybrid_proto_router_eta{eta:g}"
        hybrid_cache[eta] = proto
        results[key] = run_proto_router(key, proto, topk=None)

    for eta in etas:
        proto = hybrid_cache[eta]
        for k in topk_values:
            key = f"top{k}_hybrid_proto_router_eta{eta:g}"
            results[key] = run_proto_router(key, proto, topk=k)

    # Oracle task router upper bound.
    p_task_oracle = task_posterior_oracle_np(model, y_true)
    results["oracle_task_router"] = evaluate_router(
        model,
        "oracle_task_router",
        p_task_oracle,
        shared_logits_np,
        expert_logits_by_task,
        y_true,
    )

    # -------------------------
    # Summary
    # -------------------------
    flat_top1 = {}
    for key, value in results.items():
        if isinstance(value, dict) and "p_final" in value:
            flat_top1[key + "_p_final"] = value["p_final"]["top1"]
            flat_top1[key + "_p_moe"] = value["p_moe"]["top1"]
        elif isinstance(value, dict) and "top1" in value:
            flat_top1[key] = value["top1"]

    flat_task_acc = {
        key: value["task_acc"]
        for key, value in results.items()
        if isinstance(value, dict) and "task_acc" in value
    }

    best_name = max(flat_top1, key=flat_top1.get)

    print("\n========== Prototype Router Summary ==========")
    for key, value in flat_top1.items():
        print(f"{key:50s}: {value:.2f}")
    print(f"\nBest top1: {best_name} = {flat_top1[best_name]:.2f}")

    if flat_task_acc:
        print("\n========== Task Router Accuracy Summary ==========")
        for key, value in flat_task_acc.items():
            print(f"{key:50s}: {value:.2f}")

    if args_cli.output_json:
        output = {
            "checkpoint": args_cli.checkpoint,
            "config": args_cli.config,
            "task": target_task,
            "total_classes": int(ckpt["total_classes"]),
            "eval_known_classes": int(model._known_classes),
            "proto_scale": float(args_cli.proto_scale),
            "gaussian_scale": float(args_cli.gaussian_scale),
            "gaussian_var_floor": float(args_cli.gaussian_var_floor),
            "gaussian_shrinkage": float(args_cli.gaussian_shrinkage),
            "gaussian_use_logdet": not bool(args_cli.no_gaussian_logdet),
            "energy_mix_lambdas": mix_lambdas,
            "hybrid_etas": etas,
            "topk_values": topk_values,
            "activation_topk_values": activation_topk_values,
            "prototype_counts": {
                "min": int(mean_counts.min()),
                "max": int(mean_counts.max()),
                "missing": int((mean_counts == 0).sum()),
                "counts": mean_counts,
            },
            "results": results,
            "flat_top1": flat_top1,
            "flat_task_acc": flat_task_acc,
            "best_top1": {
                "name": best_name,
                "value": flat_top1[best_name],
            },
        }

        output_path = Path(args_cli.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(to_jsonable(output), f, indent=2)
        print(f"\nSaved summary json: {output_path}")


if __name__ == "__main__":
    main()
