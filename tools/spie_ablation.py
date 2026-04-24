import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running this script directly from tools/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import factory
from utils.data_manager import DataManager


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


def logsumexp_np(x: np.ndarray, axis=1, keepdims=False):
    max_value = np.max(x, axis=axis, keepdims=True)
    out = max_value + np.log(np.exp(x - max_value).sum(axis=axis, keepdims=True) + 1e-12)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


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


def clean_marginal_fusion_probs_np(model, shared_logits, expert_logits_by_task):
    """
    Clean posterior fusion:

        p_final(c | x)
        =
        p_shared(y in task(c) | x)
        *
        p_expert(c | x, y in task(c))

    No alpha, no temperature, no extra task-softmax.
    """
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_clean = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start_idx, end_idx) in enumerate(model.task_class_ranges):
        task_mass = p_shared[:, start_idx:end_idx].sum(axis=1, keepdims=True)
        local_prob = stable_softmax_np(expert_logits_by_task[task_id]).astype(np.float32)
        p_clean[:, start_idx:end_idx] = task_mass * local_prob

    return p_clean


def product_marginal_fusion_probs_np(model, shared_logits, expert_logits_by_task):
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_final = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start, end) in enumerate(model.task_class_ranges):
        block = p_shared[:, start:end]
        task_mass = block.sum(axis=1, keepdims=True)
        shared_local = block / np.clip(task_mass, 1e-12, None)
        expert_local = stable_softmax_np(expert_logits_by_task[task_id]).astype(np.float32)

        local = shared_local * expert_local
        local = local / np.clip(local.sum(axis=1, keepdims=True), 1e-12, None)

        p_final[:, start:end] = task_mass * local

    return p_final


def oracle_task_local_acc(model, shared_logits_np, expert_logits_by_task, y_true):
    shared_correct = 0
    expert_correct = 0
    total = len(y_true)

    for i, y in enumerate(y_true):
        true_task = None
        local_y = None
        for task_id, (start, end) in enumerate(model.task_class_ranges):
            if start <= y < end:
                true_task = task_id
                local_y = y - start
                break

        if true_task is None:
            raise ValueError(f"Target {int(y)} does not belong to any task class range.")

        start, end = model.task_class_ranges[true_task]
        shared_local_pred = np.argmax(shared_logits_np[i, start:end])
        expert_local_pred = np.argmax(expert_logits_by_task[true_task][i])

        if shared_local_pred == local_y:
            shared_correct += 1
        if expert_local_pred == local_y:
            expert_correct += 1

    return {
        "shared_oracle_task_local_top1": 100.0 * shared_correct / total,
        "expert_oracle_task_local_top1": 100.0 * expert_correct / total,
    }


def expert_ood_score_np(logits, score_type="energy"):
    """
    Larger score means the sample is more likely to be in-distribution for this expert.
    """
    if score_type == "energy":
        return logsumexp_np(logits, axis=1) - np.log(logits.shape[1])

    if score_type == "max_logit":
        return np.max(logits, axis=1)

    if score_type == "margin":
        if logits.shape[1] < 2:
            return np.zeros((logits.shape[0],), dtype=logits.dtype)
        part = np.partition(logits, kth=-2, axis=1)
        return part[:, -1] - part[:, -2]

    if score_type == "msp":
        prob = stable_softmax_np(logits)
        return np.max(prob, axis=1)

    if score_type == "neg_entropy":
        prob = stable_softmax_np(logits)
        entropy = -(prob * np.log(np.clip(prob, 1e-12, None))).sum(axis=1)
        return -entropy

    raise ValueError(f"Unknown score_type: {score_type}")


def fit_threshold_balanced_acc(id_scores, ood_scores):
    """
    ID accepts when score > tau; OOD rejects when score <= tau.
    """
    if len(id_scores) == 0 or len(ood_scores) == 0:
        return -np.inf, {
            "balanced_acc": None,
            "id_accept_rate": None,
            "ood_reject_rate": None,
        }

    all_scores = np.concatenate([id_scores, ood_scores])
    candidates = np.percentile(all_scores, np.linspace(0, 100, 501))
    candidates = np.unique(candidates)

    best_tau = candidates[0]
    best_bal_acc = -1.0
    best_id_accept = 0.0
    best_ood_reject = 0.0

    for tau in candidates:
        id_accept = (id_scores > tau).mean()
        ood_reject = (ood_scores <= tau).mean()
        bal_acc = 0.5 * (id_accept + ood_reject)

        if bal_acc > best_bal_acc:
            best_tau = tau
            best_bal_acc = bal_acc
            best_id_accept = id_accept
            best_ood_reject = ood_reject

    return float(best_tau), {
        "balanced_acc": float(best_bal_acc),
        "id_accept_rate": float(best_id_accept),
        "ood_reject_rate": float(best_ood_reject),
    }


def calibrate_future_ood_thresholds(
    model,
    expert_logits_by_task_calib,
    y_calib,
    score_type="energy",
):
    """
    For expert t, fit ID as task t train samples and OOD as later-task train samples.
    The last expert has no future OOD and is set to always accept.
    """
    tau_by_task = []
    stats_by_task = []
    task_ranges = list(model.task_class_ranges)
    num_tasks = len(task_ranges)

    for task_id, (start, end) in enumerate(task_ranges):
        logits_t = expert_logits_by_task_calib[task_id]
        scores_t = expert_ood_score_np(logits_t, score_type=score_type)

        id_mask = (y_calib >= start) & (y_calib < end)
        ood_mask = y_calib >= end

        id_scores = scores_t[id_mask]
        ood_scores = scores_t[ood_mask]

        if task_id == num_tasks - 1 or len(ood_scores) == 0:
            tau = -np.inf
            stats = {
                "task_id": task_id,
                "tau": tau,
                "num_id": int(len(id_scores)),
                "num_ood_future": int(len(ood_scores)),
                "balanced_acc": None,
                "id_accept_rate": None,
                "ood_reject_rate": None,
                "note": "last task or no future OOD; always accept",
            }
        else:
            tau, fit_stats = fit_threshold_balanced_acc(id_scores, ood_scores)
            stats = {
                "task_id": task_id,
                "tau": tau,
                "num_id": int(len(id_scores)),
                "num_ood_future": int(len(ood_scores)),
                **fit_stats,
            }

        tau_by_task.append(tau)
        stats_by_task.append(stats)

    return np.array(tau_by_task, dtype=np.float32), stats_by_task


def ood_hard_accept_fusion_probs_np(
    model,
    shared_logits,
    expert_logits_by_task,
    tau_by_task,
    score_type="energy",
    use_shared_mass=True,
):
    """
    Hard OOD-gated expert fusion. If every expert rejects a sample, fall back to
    the shared posterior for that sample.
    """
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_final = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start, end) in enumerate(model.task_class_ranges):
        expert_logits = expert_logits_by_task[task_id]
        expert_local = stable_softmax_np(expert_logits).astype(np.float32)

        scores = expert_ood_score_np(expert_logits, score_type=score_type)
        accept = (scores > tau_by_task[task_id]).astype(np.float32)[:, None]

        if use_shared_mass:
            task_mass = p_shared[:, start:end].sum(axis=1, keepdims=True)
            p_final[:, start:end] = task_mass * accept * expert_local
        else:
            p_final[:, start:end] = accept * expert_local

    row_sum = p_final.sum(axis=1, keepdims=True)
    zero_mask = row_sum[:, 0] <= 1e-12
    p_final = p_final / np.clip(row_sum, 1e-12, None)

    if np.any(zero_mask):
        p_final[zero_mask] = p_shared[zero_mask]

    return p_final


def ood_soft_accept_fusion_probs_np(
    model,
    shared_logits,
    expert_logits_by_task,
    tau_by_task,
    score_type="max_logit",
    temp=0.05,
    floor=0.2,
    use_shared_mass=True,
):
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_final = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start, end) in enumerate(model.task_class_ranges):
        expert_logits = expert_logits_by_task[task_id]
        expert_local = stable_softmax_np(expert_logits).astype(np.float32)

        scores = expert_ood_score_np(expert_logits, score_type=score_type)

        if np.isneginf(tau_by_task[task_id]):
            accept = np.ones_like(scores, dtype=np.float32)[:, None]
        else:
            raw = sigmoid_np((scores - tau_by_task[task_id]) / temp)
            accept = floor + (1.0 - floor) * raw
            accept = accept.astype(np.float32)[:, None]

        if use_shared_mass:
            task_mass = p_shared[:, start:end].sum(axis=1, keepdims=True)
            p_final[:, start:end] = task_mass * accept * expert_local
        else:
            p_final[:, start:end] = accept * expert_local

    row_sum = p_final.sum(axis=1, keepdims=True)
    p_final = p_final / np.clip(row_sum, 1e-12, None)
    return p_final


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
    Checkpoints are saved after after_task(), so known_classes often equals
    total_classes. For grouped old/new accuracy, use the current task start.
    """
    ckpt_known_classes = int(ckpt["known_classes"])
    total_classes = int(ckpt["total_classes"])
    if ckpt_known_classes >= total_classes and model.task_class_ranges:
        model._known_classes = int(model.task_class_ranges[-1][0])
    else:
        model._known_classes = ckpt_known_classes


def print_metric(name, accy):
    print(f"\n[{name}]")
    print(f"top1: {accy['top1']:.2f}")
    print(f"top5: {accy['top5']:.2f}")
    print("grouped:", accy["grouped"])


def parse_device(device_arg: str):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SPiE checkpoint ablations: shared FC, original posterior fusion, and clean marginal fusion."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the SPiE experiment json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved task_*.pkl checkpoint.")
    parser.add_argument("--device", type=str, default="0", help="CUDA id, or 'cpu'. Default: 0.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Calibration loader batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Calibration loader worker count.")
    parser.add_argument(
        "--ood-scores",
        type=str,
        default="energy,max_logit,margin,msp,neg_entropy",
        help="Comma-separated OOD score types.",
    )
    parser.add_argument(
        "--soft-ood-scores",
        type=str,
        default="max_logit,msp,margin",
        help="Comma-separated OOD score types for soft accept grid search.",
    )
    parser.add_argument(
        "--soft-ood-temps",
        type=str,
        default="0.02,0.05,0.1,0.2",
        help="Comma-separated temperatures for soft accept grid search.",
    )
    parser.add_argument(
        "--soft-ood-floors",
        type=str,
        default="0.1,0.2,0.4,0.6",
        help="Comma-separated floors for soft accept grid search.",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        type=str,
        default=None,
        help="Optional path to save summary json.",
    )
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        args = json.load(f)

    if args.get("model_name", "").lower() != "spie":
        raise ValueError(f"This script only supports model_name='spie', got {args.get('model_name')!r}.")

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
            print("  ", key)
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if unexpected:
        print("\n[Warning] Unexpected keys:")
        for key in unexpected[:20]:
            print("  ", key)
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    model._network.to(device)
    model._network.eval()

    print("\nCollecting test logits...")
    shared_logits_np, expert_logits_by_task, y_true = model._collect_eval_logits_np(
        model.test_loader
    )

    shared_pred = model._predict_topk_np(shared_logits_np)
    shared_accy = model._evaluate(shared_pred, y_true)

    p_moe, p_final = model._posterior_fusion_probs_np(
        shared_logits_np,
        expert_logits_by_task,
    )
    p_moe_pred = model._predict_topk_np(p_moe)
    p_final_pred = model._predict_topk_np(p_final)

    p_moe_accy = model._evaluate(p_moe_pred, y_true)
    p_final_accy = model._evaluate(p_final_pred, y_true)

    p_clean = clean_marginal_fusion_probs_np(
        model,
        shared_logits_np,
        expert_logits_by_task,
    )
    p_clean_pred = model._predict_topk_np(p_clean)
    p_clean_accy = model._evaluate(p_clean_pred, y_true)

    p_product = product_marginal_fusion_probs_np(
        model,
        shared_logits_np,
        expert_logits_by_task,
    )
    p_product_pred = model._predict_topk_np(p_product)
    p_product_accy = model._evaluate(p_product_pred, y_true)

    oracle_local_accy = oracle_task_local_acc(
        model,
        shared_logits_np,
        expert_logits_by_task,
        y_true,
    )

    print_metric("shared_fc", shared_accy)
    print_metric("original_p_moe", p_moe_accy)
    print_metric("original_p_final", p_final_accy)
    print_metric("clean_marginal_fusion", p_clean_accy)
    print_metric("product_marginal_fusion", p_product_accy)

    print("\n[oracle_task_local]")
    for key, value in oracle_local_accy.items():
        print(f"{key}: {value:.2f}")

    all_results = {
        "shared_fc": shared_accy["top1"],
        "original_p_moe": p_moe_accy["top1"],
        "original_p_final": p_final_accy["top1"],
        "clean_marginal_fusion": p_clean_accy["top1"],
        "product_marginal_fusion": p_product_accy["top1"],
    }
    metric_results = {
        "shared_fc": shared_accy,
        "original_p_moe": p_moe_accy,
        "original_p_final": p_final_accy,
        "clean_marginal_fusion": p_clean_accy,
        "product_marginal_fusion": p_product_accy,
    }

    row_sum = p_clean.sum(axis=1)
    print(
        "\nClean fusion probability sum:",
        f"min={row_sum.min():.6f}",
        f"max={row_sum.max():.6f}",
        f"mean={row_sum.mean():.6f}",
    )

    product_row_sum = p_product.sum(axis=1)
    print(
        "Product fusion probability sum:",
        f"min={product_row_sum.min():.6f}",
        f"max={product_row_sum.max():.6f}",
        f"mean={product_row_sum.mean():.6f}",
    )

    print("\nCollecting calibration logits on seen training data...")
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
    calib_dataset = data_manager.get_dataset(
        np.arange(0, model._total_classes),
        source="train",
        mode="test",
    )
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    _, expert_logits_by_task_calib, y_calib = model._collect_eval_logits_np(calib_loader)

    ood_score_types = parse_csv_strings(args_cli.ood_scores)
    soft_ood_score_types = set(parse_csv_strings(args_cli.soft_ood_scores))
    calibration_score_types = list(dict.fromkeys([*ood_score_types, *soft_ood_score_types]))
    soft_ood_temps = parse_csv_floats(args_cli.soft_ood_temps)
    soft_ood_floors = parse_csv_floats(args_cli.soft_ood_floors)
    ood_calibration = {}
    ood_probability_sum = {}
    soft_ood_grid = {}

    for score_type in calibration_score_types:
        tau_by_task, ood_stats = calibrate_future_ood_thresholds(
            model,
            expert_logits_by_task_calib,
            y_calib,
            score_type=score_type,
        )

        print(f"\n[OOD calibration: {score_type}]")
        for stats in ood_stats:
            bal_acc = stats["balanced_acc"]
            id_accept = stats["id_accept_rate"]
            ood_reject = stats["ood_reject_rate"]
            print(
                f"task {stats['task_id']:02d} | "
                f"tau={stats['tau']} | "
                f"ID={stats['num_id']} | "
                f"future_OOD={stats['num_ood_future']} | "
                f"bal_acc={None if bal_acc is None else round(bal_acc, 4)} | "
                f"id_accept={None if id_accept is None else round(id_accept, 4)} | "
                f"ood_reject={None if ood_reject is None else round(ood_reject, 4)}"
            )

        ood_calibration[score_type] = {
            "tau_by_task": tau_by_task,
            "stats": ood_stats,
        }

        if score_type in ood_score_types:
            p_ood_shared_mass = ood_hard_accept_fusion_probs_np(
                model,
                shared_logits_np,
                expert_logits_by_task,
                tau_by_task,
                score_type=score_type,
                use_shared_mass=True,
            )
            pred_ood_shared_mass = model._predict_topk_np(p_ood_shared_mass)
            acc_ood_shared_mass = model._evaluate(pred_ood_shared_mass, y_true)
            name_shared_mass = f"ood_{score_type}_shared_mass_accept_expert"
            print_metric(name_shared_mass, acc_ood_shared_mass)
            all_results[name_shared_mass] = acc_ood_shared_mass["top1"]
            metric_results[name_shared_mass] = acc_ood_shared_mass

            p_ood_expert_only = ood_hard_accept_fusion_probs_np(
                model,
                shared_logits_np,
                expert_logits_by_task,
                tau_by_task,
                score_type=score_type,
                use_shared_mass=False,
            )
            pred_ood_expert_only = model._predict_topk_np(p_ood_expert_only)
            acc_ood_expert_only = model._evaluate(pred_ood_expert_only, y_true)
            name_expert_only = f"ood_{score_type}_accept_expert_only"
            print_metric(name_expert_only, acc_ood_expert_only)
            all_results[name_expert_only] = acc_ood_expert_only["top1"]
            metric_results[name_expert_only] = acc_ood_expert_only

            ood_row_sum = p_ood_shared_mass.sum(axis=1)
            ood_probability_sum[score_type] = {
                "shared_mass": {
                    "min": float(ood_row_sum.min()),
                    "max": float(ood_row_sum.max()),
                    "mean": float(ood_row_sum.mean()),
                }
            }
            print(
                f"OOD fusion probability sum ({score_type}, shared_mass):",
                f"min={ood_row_sum.min():.6f}",
                f"max={ood_row_sum.max():.6f}",
                f"mean={ood_row_sum.mean():.6f}",
            )

        if score_type in soft_ood_score_types:
            print(f"\n[OOD soft accept grid: {score_type}]")
            soft_ood_grid[score_type] = []

            for temp in soft_ood_temps:
                for floor in soft_ood_floors:
                    p_soft = ood_soft_accept_fusion_probs_np(
                        model,
                        shared_logits_np,
                        expert_logits_by_task,
                        tau_by_task,
                        score_type=score_type,
                        temp=temp,
                        floor=floor,
                        use_shared_mass=True,
                    )
                    pred_soft = model._predict_topk_np(p_soft)
                    acc_soft = model._evaluate(pred_soft, y_true)
                    name_soft = f"ood_soft_{score_type}_temp{temp:g}_floor{floor:g}_shared_mass"

                    print(f"{score_type} temp={temp:g} floor={floor:g} top1={acc_soft['top1']:.2f}")
                    all_results[name_soft] = acc_soft["top1"]
                    metric_results[name_soft] = acc_soft
                    soft_ood_grid[score_type].append(
                        {
                            "name": name_soft,
                            "score_type": score_type,
                            "temp": temp,
                            "floor": floor,
                            "top1": acc_soft["top1"],
                            "top5": acc_soft["top5"],
                        }
                    )

    best_name = max(all_results, key=all_results.get)

    print("\n========== Summary ==========")
    for key, value in all_results.items():
        print(f"{key:45s}: {value:.2f}")
    print(f"\nBest top1: {best_name} = {all_results[best_name]:.2f}")

    if args_cli.output_json:
        output = {
            "checkpoint": args_cli.checkpoint,
            "config": args_cli.config,
            "task": target_task,
            "total_classes": int(ckpt["total_classes"]),
            "eval_known_classes": int(model._known_classes),
            "top1": all_results,
            "metrics": metric_results,
            "oracle_task_local": oracle_local_accy,
            "best_top1": {"name": best_name, "value": all_results[best_name]},
            "clean_probability_sum": {
                "min": float(row_sum.min()),
                "max": float(row_sum.max()),
                "mean": float(row_sum.mean()),
            },
            "product_probability_sum": {
                "min": float(product_row_sum.min()),
                "max": float(product_row_sum.max()),
                "mean": float(product_row_sum.mean()),
            },
            "ood_calibration": ood_calibration,
            "ood_probability_sum": ood_probability_sum,
            "soft_ood_grid": soft_ood_grid,
        }
        output_path = Path(args_cli.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(to_jsonable(output), f, indent=2)
        print(f"\nSaved summary json: {output_path}")


if __name__ == "__main__":
    main()
