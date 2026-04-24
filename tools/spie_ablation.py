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


def print_metric(name, accy):
    print(f"\n[{name}]")
    print(f"top1: {accy['top1']:.2f}")
    print(f"top5: {accy['top5']:.2f}")
    print("grouped:", accy["grouped"])


def parse_device(device_arg: str):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def get_task_ranges(model):
    return list(model.task_class_ranges)


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
    Checkpoints are saved after after_task(), so known_classes often equals
    total_classes. For grouped old/new accuracy, use the current task start.
    """
    ckpt_known_classes = int(ckpt["known_classes"])
    total_classes = int(ckpt["total_classes"])

    if ckpt_known_classes >= total_classes and model.task_class_ranges:
        model._known_classes = int(model.task_class_ranges[-1][0])
    else:
        model._known_classes = ckpt_known_classes


# -------------------------
# Fusion baselines
# -------------------------

def product_marginal_fusion_probs_np(model, shared_logits, expert_logits_by_task):
    """
    Marginal-preserving product fusion:

        p(c | x)
        =
        p_s(task(c) | x)
        *
        normalize_task(
            p_s(c | x, task(c)) * p_e(c | x, task(c))
        )

    This preserves shared task marginal and only refines within-task distribution.
    """
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_final = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start, end) in enumerate(get_task_ranges(model)):
        block = p_shared[:, start:end]
        task_mass = block.sum(axis=1, keepdims=True)

        shared_local = block / np.clip(task_mass, 1e-12, None)
        expert_local = stable_softmax_np(expert_logits_by_task[task_id]).astype(np.float32)

        local = shared_local * expert_local
        local = local / np.clip(local.sum(axis=1, keepdims=True), 1e-12, None)

        p_final[:, start:end] = task_mass * local

    return p_final


# -------------------------
# Oracle local diagnosis
# -------------------------

def oracle_task_local_acc(model, shared_logits_np, expert_logits_by_task, y_true):
    shared_correct = 0
    expert_correct = 0
    total = len(y_true)

    for i, y in enumerate(y_true):
        true_task = None
        local_y = None

        for task_id, (start, end) in enumerate(get_task_ranges(model)):
            if start <= y < end:
                true_task = task_id
                local_y = y - start
                break

        if true_task is None:
            raise ValueError(f"Target {int(y)} does not belong to any task class range.")

        start, end = get_task_ranges(model)[true_task]

        shared_local_pred = np.argmax(shared_logits_np[i, start:end])
        expert_local_pred = np.argmax(expert_logits_by_task[true_task][i])

        shared_correct += int(shared_local_pred == local_y)
        expert_correct += int(expert_local_pred == local_y)

    return {
        "shared_oracle_task_local_top1": 100.0 * shared_correct / total,
        "expert_oracle_task_local_top1": 100.0 * expert_correct / total,
    }


# -------------------------
# Expert OOD / reject score
# -------------------------

def expert_ood_score_np(logits, score_type="max_logit"):
    """
    Larger score means the sample is more likely to be in-distribution for this expert.

    Kept scores:
      max_logit : max logit
      margin    : top1 logit - top2 logit
      msp       : max softmax probability
    """
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

    raise ValueError(f"Unknown score_type: {score_type}")


def fit_safe_reject_thresholds(
    model,
    expert_logits_by_task_calib,
    y_calib,
    score_type="max_logit",
    max_id_false_reject=0.01,
):
    """
    Safe reject threshold.

    Score larger means more ID-like.
    Reject rule:
        reject if score <= tau

    We choose tau as the low percentile of ID scores, so that calibration ID
    false reject rate is approximately max_id_false_reject.

    For expert t:
      ID  = task t training samples
      future OOD = task t+1 ... final task training samples

    The threshold is chosen only from ID quantile. Future OOD is used only
    for diagnostic reporting.
    """
    tau_by_task = []
    stats_by_task = []

    task_ranges = get_task_ranges(model)
    num_tasks = len(task_ranges)

    percentile = 100.0 * max_id_false_reject

    for task_id, (start, end) in enumerate(task_ranges):
        logits_t = expert_logits_by_task_calib[task_id]
        scores_t = expert_ood_score_np(logits_t, score_type=score_type)

        id_mask = (y_calib >= start) & (y_calib < end)
        future_ood_mask = y_calib >= end

        id_scores = scores_t[id_mask]
        future_ood_scores = scores_t[future_ood_mask]

        if len(id_scores) == 0 or task_id == num_tasks - 1:
            tau = -np.inf
        else:
            tau = float(np.percentile(id_scores, percentile))

        if np.isneginf(tau):
            id_false_reject = None
            future_ood_reject = None
        else:
            id_false_reject = float((id_scores <= tau).mean())
            future_ood_reject = (
                None
                if len(future_ood_scores) == 0
                else float((future_ood_scores <= tau).mean())
            )

        stats_by_task.append(
            {
                "task_id": int(task_id),
                "tau": tau,
                "num_id": int(len(id_scores)),
                "num_future_ood": int(len(future_ood_scores)),
                "id_false_reject": id_false_reject,
                "future_ood_reject": future_ood_reject,
            }
        )
        tau_by_task.append(tau)

    return np.array(tau_by_task, dtype=np.float32), stats_by_task


# -------------------------
# Semi-guaranteed veto analysis
# -------------------------

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


def analyze_top1_reject_precision(
    model,
    base_probs,
    expert_logits_by_task,
    y_true,
    tau_by_task,
    score_type="max_logit",
    name="product",
):
    """
    Analyze whether the top1 task/class from base_probs can be safely vetoed.

    Important metrics:
      false_reject_on_correct_top1_class:
          base top1 class was correct, but veto rejects it.

      false_reject_on_correct_top1_task:
          base top1 task was correct, but veto rejects that task.

      reject_precision_task_wrong:
          among rejected top1 samples, how many top1 tasks were actually wrong.

      reject_recall_on_task_wrong_top1:
          among task-wrong top1 samples, how many are caught by reject.
    """
    n = len(y_true)

    top1 = np.argmax(base_probs, axis=1)
    true_class = y_true

    true_task = class_to_task_ids(model, true_class)
    top1_task = class_to_task_ids(model, top1)

    top1_class_correct = top1 == true_class
    top1_task_correct = top1_task == true_task
    top1_task_wrong = ~top1_task_correct

    reject_top1 = np.zeros(n, dtype=bool)
    top1_score = np.zeros(n, dtype=np.float32)

    for task_id, (start, end) in enumerate(get_task_ranges(model)):
        idx = np.where(top1_task == task_id)[0]
        if len(idx) == 0:
            continue

        logits_t = expert_logits_by_task[task_id][idx]
        scores_t = expert_ood_score_np(logits_t, score_type=score_type)

        top1_score[idx] = scores_t

        tau = tau_by_task[task_id]
        if np.isneginf(tau):
            reject_top1[idx] = False
        else:
            reject_top1[idx] = scores_t <= tau

    num_reject = int(reject_top1.sum())

    def rate(event, base):
        denom = int(base.sum())
        if denom == 0:
            return None
        return float((event & base).sum() / denom)

    stats = {
        "base_name": name,
        "score_type": score_type,
        "num_samples": int(n),

        "base_top1_acc": float(top1_class_correct.mean()),
        "base_top1_task_acc": float(top1_task_correct.mean()),

        "top1_reject_rate": float(reject_top1.mean()),
        "num_top1_rejected": num_reject,

        "false_reject_on_correct_top1_class": rate(reject_top1, top1_class_correct),
        "false_reject_on_correct_top1_task": rate(reject_top1, top1_task_correct),

        "reject_precision_class_wrong": (
            None
            if num_reject == 0
            else float((reject_top1 & ~top1_class_correct).sum() / num_reject)
        ),
        "reject_precision_task_wrong": (
            None
            if num_reject == 0
            else float((reject_top1 & top1_task_wrong).sum() / num_reject)
        ),

        "reject_recall_on_task_wrong_top1": rate(reject_top1, top1_task_wrong),

        "num_false_reject_correct_class": int((reject_top1 & top1_class_correct).sum()),
        "num_false_reject_correct_task": int((reject_top1 & top1_task_correct).sum()),
        "num_reject_task_wrong": int((reject_top1 & top1_task_wrong).sum()),
    }

    print(f"\n[top1_reject_precision | base={name} | score={score_type}]")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return stats


def semi_veto_predict_from_base_probs(
    model,
    base_probs,
    expert_logits_by_task,
    tau_by_task,
    score_type="max_logit",
    topk=5,
):
    """
    Conservative semi-veto inference.

    1. Use base_probs to rank all classes.
    2. Visit candidates in descending score order.
    3. If the candidate's task expert rejects this sample, skip the whole task.
    4. Otherwise keep the class.
    5. If not enough candidates remain, fill from the original base ranking.

    This does NOT add/subtract scores. It only vetoes a task when its expert
    safely rejects the sample.
    """
    n, _ = base_probs.shape
    task_ranges = get_task_ranges(model)

    preds = np.zeros((n, topk), dtype=np.int64)

    # Precompute per-sample per-task reject decision.
    reject_task = np.zeros((n, len(task_ranges)), dtype=bool)

    for task_id, (start, end) in enumerate(task_ranges):
        logits_t = expert_logits_by_task[task_id]
        scores_t = expert_ood_score_np(logits_t, score_type=score_type)
        tau = tau_by_task[task_id]

        if np.isneginf(tau):
            reject_task[:, task_id] = False
        else:
            reject_task[:, task_id] = scores_t <= tau

    rejected_any = np.zeros(n, dtype=bool)
    fallback = np.zeros(n, dtype=bool)
    num_skipped_tasks = np.zeros(n, dtype=np.int64)

    for i in range(n):
        ranked = np.argsort(-base_probs[i])

        skipped_tasks = set()
        candidates = []

        for c in ranked:
            c = int(c)

            task_id = None
            for t, (start, end) in enumerate(task_ranges):
                if start <= c < end:
                    task_id = t
                    break

            if task_id is None:
                continue

            if task_id in skipped_tasks:
                continue

            if reject_task[i, task_id]:
                skipped_tasks.add(task_id)
                rejected_any[i] = True
                continue

            candidates.append(c)

            if len(candidates) >= topk:
                break

        if len(candidates) < topk:
            fallback[i] = True
            for c in ranked:
                c = int(c)
                if c not in candidates:
                    candidates.append(c)
                if len(candidates) >= topk:
                    break

        num_skipped_tasks[i] = len(skipped_tasks)
        preds[i] = np.array(candidates[:topk], dtype=np.int64)

    diag = {
        "rejected_any_rate": float(rejected_any.mean()),
        "fallback_rate": float(fallback.mean()),
        "avg_skipped_tasks": float(num_skipped_tasks.mean()),
    }

    return preds, diag


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SPiE semi-veto ablation: product fusion + conservative expert rejection."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the SPiE experiment json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved task_*.pkl checkpoint.")
    parser.add_argument("--device", type=str, default="0", help="CUDA id, or 'cpu'. Default: 0.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Calibration loader batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Calibration loader worker count.")

    parser.add_argument(
        "--reject-scores",
        type=str,
        default="max_logit,msp,margin",
        help="Comma-separated reject score types. Recommended: max_logit,msp,margin.",
    )
    parser.add_argument(
        "--id-false-reject-rates",
        type=str,
        default="0.001,0.005,0.01,0.02",
        help="Comma-separated safe ID false reject rates for threshold selection.",
    )
    parser.add_argument(
        "--base-for-veto",
        type=str,
        default="product",
        choices=["product", "shared", "original_p_final"],
        help="Which base ranking to apply semi-veto to.",
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

    # -------------------------
    # Collect test logits
    # -------------------------

    print("\nCollecting test logits...")
    shared_logits_np, expert_logits_by_task, y_true = model._collect_eval_logits_np(
        model.test_loader
    )

    # -------------------------
    # Baselines
    # -------------------------

    shared_probs = stable_softmax_np(shared_logits_np).astype(np.float32)
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
    print_metric("product_marginal_fusion", p_product_accy)

    print("\n[oracle_task_local]")
    for key, value in oracle_local_accy.items():
        print(f"{key}: {value:.2f}")

    product_row_sum = p_product.sum(axis=1)
    print(
        "\nProduct fusion probability sum:",
        f"min={product_row_sum.min():.6f}",
        f"max={product_row_sum.max():.6f}",
        f"mean={product_row_sum.mean():.6f}",
    )

    all_results = {
        "shared_fc": shared_accy["top1"],
        "original_p_moe": p_moe_accy["top1"],
        "original_p_final": p_final_accy["top1"],
        "product_marginal_fusion": p_product_accy["top1"],
    }

    metric_results = {
        "shared_fc": shared_accy,
        "original_p_moe": p_moe_accy,
        "original_p_final": p_final_accy,
        "product_marginal_fusion": p_product_accy,
    }

    if args_cli.base_for_veto == "product":
        base_probs = p_product
        base_name = "product"
    elif args_cli.base_for_veto == "shared":
        base_probs = shared_probs
        base_name = "shared"
    elif args_cli.base_for_veto == "original_p_final":
        base_probs = p_final
        base_name = "original_p_final"
    else:
        raise ValueError(args_cli.base_for_veto)

    # -------------------------
    # Collect calibration logits
    # -------------------------

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

    # -------------------------
    # Semi-veto experiments
    # -------------------------

    reject_score_types = parse_csv_strings(args_cli.reject_scores)
    id_false_reject_rates = parse_csv_floats(args_cli.id_false_reject_rates)

    semi_veto_diagnostics = {}
    safe_thresholds = {}

    for score_type in reject_score_types:
        for idfr in id_false_reject_rates:
            tau_by_task, threshold_stats = fit_safe_reject_thresholds(
                model,
                expert_logits_by_task_calib,
                y_calib,
                score_type=score_type,
                max_id_false_reject=idfr,
            )

            key = f"{score_type}_idfr{idfr:g}"
            safe_thresholds[key] = {
                "score_type": score_type,
                "id_false_reject_rate": idfr,
                "tau_by_task": tau_by_task,
                "stats": threshold_stats,
            }

            print(
                f"\n[SAFE reject thresholds | score={score_type} | "
                f"target_id_false_reject={idfr:g}]"
            )
            for st in threshold_stats:
                print(
                    f"task {st['task_id']:02d} | "
                    f"tau={st['tau']} | "
                    f"ID={st['num_id']} | "
                    f"future_OOD={st['num_future_ood']} | "
                    f"id_false_reject={st['id_false_reject']} | "
                    f"future_ood_reject={st['future_ood_reject']}"
                )

            reject_stats = analyze_top1_reject_precision(
                model,
                base_probs,
                expert_logits_by_task,
                y_true,
                tau_by_task,
                score_type=score_type,
                name=base_name,
            )

            veto_pred, veto_diag = semi_veto_predict_from_base_probs(
                model,
                base_probs,
                expert_logits_by_task,
                tau_by_task,
                score_type=score_type,
                topk=5,
            )

            veto_accy = model._evaluate(veto_pred, y_true)
            veto_name = f"semi_veto_{base_name}_{score_type}_idfr{idfr:g}"

            print_metric(veto_name, veto_accy)
            print(f"[{veto_name} diag]")
            for k, v in veto_diag.items():
                print(f"{k}: {v:.4f}")

            all_results[veto_name] = veto_accy["top1"]
            metric_results[veto_name] = veto_accy

            semi_veto_diagnostics[veto_name] = {
                "reject_stats": reject_stats,
                "veto_diag": veto_diag,
            }

    # -------------------------
    # Summary
    # -------------------------

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
            "base_for_veto": base_name,
            "top1": all_results,
            "metrics": metric_results,
            "oracle_task_local": oracle_local_accy,
            "best_top1": {
                "name": best_name,
                "value": all_results[best_name],
            },
            "product_probability_sum": {
                "min": float(product_row_sum.min()),
                "max": float(product_row_sum.max()),
                "mean": float(product_row_sum.mean()),
            },
            "safe_thresholds": safe_thresholds,
            "semi_veto_diagnostics": semi_veto_diagnostics,
        }

        output_path = Path(args_cli.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(to_jsonable(output), f, indent=2)

        print(f"\nSaved summary json: {output_path}")


if __name__ == "__main__":
    main()