import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import load_json
from trainer import _set_device, _set_random, print_args
from utils import factory
from utils.data_manager import DataManager


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Post-hoc calibration and merged evaluation for SPiE v14 checkpoints."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument(
        "--max-calib-per-class",
        type=int,
        default=None,
        help="Optional cap for calibration samples per class from the train split.",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=None,
        help="Optional cap for final test samples per class.",
    )
    parser.add_argument(
        "--tau-strategy",
        type=str,
        default="prefer_tpr95",
        choices=["prefer_tpr95", "quantile_only"],
        help="How to compute tau_t.",
    )
    parser.add_argument(
        "--tau-quantile",
        type=float,
        default=0.05,
        help="Fallback quantile for tau_t when TPR95 threshold is unavailable.",
    )
    parser.add_argument("--sigma-eps", type=float, default=1e-8, help="Minimum sigma stabilizer.")
    parser.add_argument("--note", type=str, default="posthoc-calibration", help="Optional logging note.")
    return parser


def _subset_dataset_per_class(dataset, max_per_class):
    if max_per_class is None:
        return dataset

    label_to_indices = defaultdict(list)
    labels = np.asarray(dataset.labels)
    for idx, label in enumerate(labels):
        label = int(label)
        if len(label_to_indices[label]) < max_per_class:
            label_to_indices[label].append(idx)

    subset_indices = []
    for label in sorted(label_to_indices.keys()):
        subset_indices.extend(label_to_indices[label])
    return Subset(dataset, subset_indices)


def _make_loader(dataset, batch_size, num_workers, max_per_class=None, shuffle=False):
    dataset = _subset_dataset_per_class(dataset, max_per_class)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _materialize_detector_buffers_from_checkpoint(model, state_dict):
    for expert_id in range(len(model._network.task_ood_detectors)):
        detector = model._network.get_task_ood_detector(expert_id)
        for buffer_name in [
            "class_centers",
            "class_diag_vars",
            "representatives",
            "positive_bank",
            "ood_bank",
        ]:
            key = f"task_ood_detectors.{expert_id}.{buffer_name}"
            tensor = state_dict.get(key)
            if tensor is None:
                continue
            setattr(
                detector,
                buffer_name,
                tensor.detach().clone().to(device=model._device, dtype=torch.float32),
            )


def _rebuild_spie_v14_from_checkpoint(model, data_manager, checkpoint):
    if model.args["model_name"] != "spie_v14":
        raise ValueError("This evaluator currently only supports model_name='spie_v14'.")

    task_id = int(checkpoint["tasks"])
    if task_id < 0:
        raise ValueError(f"Invalid checkpoint task id: {task_id}")

    model._cur_task = task_id
    model._total_classes = int(checkpoint["total_classes"])
    model._known_classes = model._total_classes
    model.task_class_ranges = []

    start = 0
    backbone = model._backbone_module()
    for cur_task in range(task_id + 1):
        task_size = int(data_manager.get_task_size(cur_task))
        end = start + task_size
        model._network.update_fc(task_size)
        model.task_class_ranges.append((start, end))
        backbone.reset_task_modules()
        backbone.adapter_update()
        start = end

    _materialize_detector_buffers_from_checkpoint(model, checkpoint["model_state_dict"])
    missing, unexpected = model._network.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint structure mismatch. Missing={missing}, unexpected={unexpected}")

    model._network.to(model._device)
    model._network.eval()
    return task_id


def _binary_clf_curve(y_true, y_score):
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[order]
    y_score = y_score[order]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    return fps.astype(np.float64), tps.astype(np.float64), thresholds.astype(np.float64)


def _compute_fpr95(y_true, y_score, target_tpr=0.95):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_count = int(y_true.sum())
    neg_count = int((1 - y_true).sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan"), float("nan")

    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    fpr = fps / neg_count
    tpr = tps / pos_count
    meets_target = np.where(tpr >= target_tpr)[0]
    if meets_target.size == 0:
        return float("nan"), float("nan")

    idx = int(meets_target[0])
    return float(fpr[idx]), float(thresholds[idx])


def _class_to_task_id(task_ranges, class_idx):
    for task_id, (start, end) in enumerate(task_ranges):
        if start <= class_idx < end:
            return task_id
    raise ValueError(f"Class {class_idx} is not covered by any task range.")


@torch.no_grad()
def _collect_scores_for_expert(model, loader, expert_id):
    detector = model._network.get_task_ood_detector(expert_id)
    score_chunks = []

    for _, inputs, _targets in loader:
        inputs = inputs.to(model._device)
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["expert_features"]
        scores = detector.score_from_stats(detector.compute_stats(features))
        score_chunks.append(scores.detach().cpu())

    if not score_chunks:
        return np.zeros(0, dtype=np.float32)
    return torch.cat(score_chunks, dim=0).numpy()


def _safe_round(value, digits=6):
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return round(float(value), digits)


def _counter_to_sorted_dict(counter):
    return {str(key): int(counter[key]) for key in sorted(counter.keys())}


def _build_task_loaders(data_manager, total_classes, batch_size, num_workers, max_calib_per_class):
    task_loaders = []
    class_cursor = 0
    for task_id in range(data_manager.nb_tasks):
        task_size = data_manager.get_task_size(task_id)
        task_end = class_cursor + task_size
        if class_cursor >= total_classes:
            break
        indices = np.arange(class_cursor, min(task_end, total_classes))
        task_dataset = data_manager.get_dataset(indices, source="train", mode="test")
        task_loader = _make_loader(
            task_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            max_per_class=max_calib_per_class,
            shuffle=False,
        )
        task_loaders.append(task_loader)
        class_cursor = task_end
    return task_loaders


def _compute_calibration_parameters(model, task_loaders, tau_strategy, tau_quantile, sigma_eps):
    active_expert_ids = model._active_expert_ids()
    calibration = []

    for expert_id in active_expert_ids:
        id_scores = _collect_scores_for_expert(model, task_loaders[expert_id], expert_id)
        future_score_chunks = []
        for future_task in range(expert_id + 1, len(active_expert_ids)):
            future_scores = _collect_scores_for_expert(model, task_loaders[future_task], expert_id)
            if future_scores.size > 0:
                future_score_chunks.append(future_scores)
        ood_scores = (
            np.concatenate(future_score_chunks, axis=0).astype(np.float32, copy=False)
            if future_score_chunks
            else np.zeros(0, dtype=np.float32)
        )

        threshold95 = None
        fpr95 = None
        if tau_strategy == "prefer_tpr95" and id_scores.size > 0 and ood_scores.size > 0:
            labels = np.concatenate(
                [
                    np.ones(id_scores.shape[0], dtype=np.int64),
                    np.zeros(ood_scores.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            scores = np.concatenate([id_scores, ood_scores], axis=0)
            fpr95, threshold95 = _compute_fpr95(labels, scores, target_tpr=0.95)
            if np.isnan(threshold95):
                threshold95 = None
            if np.isnan(fpr95):
                fpr95 = None

        quantile_tau = float(np.quantile(id_scores, tau_quantile)) if id_scores.size > 0 else 0.0
        tau = float(threshold95) if threshold95 is not None else quantile_tau

        if id_scores.size > 1:
            sigma = float(np.std(id_scores, ddof=0))
        elif id_scores.size == 1:
            sigma = float(abs(id_scores[0]))
        else:
            sigma = 0.0
        sigma = max(sigma, float(sigma_eps))

        detector = model._network.get_task_ood_detector(expert_id)
        calibration.append(
            {
                "expert_id": int(expert_id),
                "task_range": list(model.task_class_ranges[expert_id]),
                "num_id": int(id_scores.shape[0]),
                "num_future_ood": int(ood_scores.shape[0]),
                "stored_ood_bank_size": int(detector.ood_bank.shape[0]),
                "tau": _safe_round(tau),
                "tau_source": "threshold_at_tpr95" if threshold95 is not None else f"id_q{tau_quantile:.2f}",
                "sigma": _safe_round(sigma),
                "sigma_raw": float(sigma),
                "threshold_at_tpr95": _safe_round(threshold95),
                "fpr95": _safe_round(fpr95),
                "id_score_mean": _safe_round(float(id_scores.mean()) if id_scores.size > 0 else None),
                "id_score_std": _safe_round(float(np.std(id_scores, ddof=0)) if id_scores.size > 0 else None),
                "id_score_q05": _safe_round(float(np.quantile(id_scores, 0.05)) if id_scores.size > 0 else None),
                "id_score_q50": _safe_round(float(np.quantile(id_scores, 0.50)) if id_scores.size > 0 else None),
                "ood_score_mean": _safe_round(float(ood_scores.mean()) if ood_scores.size > 0 else None),
                "ood_score_q50": _safe_round(float(np.quantile(ood_scores, 0.50)) if ood_scores.size > 0 else None),
            }
        )

    tau_tensor = torch.tensor([item["tau"] for item in calibration], device=model._device, dtype=torch.float32)
    sigma_tensor = torch.tensor([item["sigma_raw"] for item in calibration], device=model._device, dtype=torch.float32)
    return calibration, tau_tensor, sigma_tensor


def _init_eval_stats(num_tasks):
    return {
        "correct": 0,
        "total": 0,
        "old_correct": 0,
        "old_total": 0,
        "new_correct": 0,
        "new_total": 0,
        "per_task_correct": [0 for _ in range(num_tasks)],
        "per_task_total": [0 for _ in range(num_tasks)],
        "wrong_argmax_task_counter": Counter(),
        "wrong_argmax_later_counter": Counter(),
        "wrong_total": 0,
        "wrong_to_later_task": 0,
        "wrong_to_earlier_task": 0,
        "wrong_to_same_task": 0,
    }


def _finalize_eval_stats(stats):
    total = max(stats["total"], 1)
    old_total = max(stats["old_total"], 1)
    new_total = max(stats["new_total"], 1)
    wrong_total = max(stats["wrong_total"], 1)

    return {
        "top1": _safe_round(stats["correct"] * 100.0 / total, digits=4),
        "old_top1": _safe_round(stats["old_correct"] * 100.0 / old_total, digits=4) if stats["old_total"] > 0 else None,
        "new_top1": _safe_round(stats["new_correct"] * 100.0 / new_total, digits=4) if stats["new_total"] > 0 else None,
        "per_task_top1": {
            str(task_id): _safe_round(
                stats["per_task_correct"][task_id] * 100.0 / stats["per_task_total"][task_id],
                digits=4,
            )
            if stats["per_task_total"][task_id] > 0
            else None
            for task_id in range(len(stats["per_task_total"]))
        },
        "wrong_argmax_task_hist": _counter_to_sorted_dict(stats["wrong_argmax_task_counter"]),
        "wrong_argmax_later_task_hist": _counter_to_sorted_dict(stats["wrong_argmax_later_counter"]),
        "wrong_total": int(stats["wrong_total"]),
        "wrong_to_later_task": int(stats["wrong_to_later_task"]),
        "wrong_to_later_task_rate": _safe_round(stats["wrong_to_later_task"] * 100.0 / wrong_total, digits=4)
        if stats["wrong_total"] > 0
        else None,
        "wrong_to_earlier_task": int(stats["wrong_to_earlier_task"]),
        "wrong_to_same_task": int(stats["wrong_to_same_task"]),
    }


@torch.no_grad()
def _evaluate_baseline_and_calibrated(
    model,
    loader,
    tau_tensor,
    sigma_tensor,
):
    active_expert_ids = model._active_expert_ids()
    num_tasks = len(active_expert_ids)
    old_task_boundary = model.task_class_ranges[-1][0]

    baseline_stats = _init_eval_stats(num_tasks)
    calibrated_stats = _init_eval_stats(num_tasks)

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)

        shared_logits = model._shared_cls_logits(inputs)
        ood_out = model._network.forward_multi_expert_ood_scores(inputs, active_expert_ids)
        raw_scores = ood_out["scores"]
        calibrated_scores = (raw_scores - tau_tensor.unsqueeze(1)) / (sigma_tensor.unsqueeze(1) + 1e-8)

        baseline_weights = model._compute_task_prior_weights(raw_scores)
        calibrated_weights = model._compute_task_prior_weights(calibrated_scores)

        baseline_logits = model._apply_task_prior(shared_logits, baseline_weights, active_expert_ids)
        calibrated_logits = model._apply_task_prior(shared_logits, calibrated_weights, active_expert_ids)

        baseline_preds = torch.argmax(baseline_logits, dim=1)
        calibrated_preds = torch.argmax(calibrated_logits, dim=1)

        baseline_argmax_tasks = torch.argmax(baseline_weights, dim=0)
        calibrated_argmax_tasks = torch.argmax(calibrated_weights, dim=0)

        for sample_idx in range(targets.shape[0]):
            target = int(targets[sample_idx].item())
            true_task_id = _class_to_task_id(model.task_class_ranges, target)
            is_old = target < old_task_boundary

            baseline_stats["total"] += 1
            calibrated_stats["total"] += 1
            baseline_stats["per_task_total"][true_task_id] += 1
            calibrated_stats["per_task_total"][true_task_id] += 1

            if is_old:
                baseline_stats["old_total"] += 1
                calibrated_stats["old_total"] += 1
            else:
                baseline_stats["new_total"] += 1
                calibrated_stats["new_total"] += 1

            baseline_pred = int(baseline_preds[sample_idx].item())
            calibrated_pred = int(calibrated_preds[sample_idx].item())
            baseline_argmax_task = int(baseline_argmax_tasks[sample_idx].item())
            calibrated_argmax_task = int(calibrated_argmax_tasks[sample_idx].item())

            if baseline_pred == target:
                baseline_stats["correct"] += 1
                baseline_stats["per_task_correct"][true_task_id] += 1
                if is_old:
                    baseline_stats["old_correct"] += 1
                else:
                    baseline_stats["new_correct"] += 1
            else:
                baseline_stats["wrong_total"] += 1
                baseline_stats["wrong_argmax_task_counter"][baseline_argmax_task] += 1
                if baseline_argmax_task > true_task_id:
                    baseline_stats["wrong_to_later_task"] += 1
                    baseline_stats["wrong_argmax_later_counter"][baseline_argmax_task] += 1
                elif baseline_argmax_task < true_task_id:
                    baseline_stats["wrong_to_earlier_task"] += 1
                else:
                    baseline_stats["wrong_to_same_task"] += 1

            if calibrated_pred == target:
                calibrated_stats["correct"] += 1
                calibrated_stats["per_task_correct"][true_task_id] += 1
                if is_old:
                    calibrated_stats["old_correct"] += 1
                else:
                    calibrated_stats["new_correct"] += 1
            else:
                calibrated_stats["wrong_total"] += 1
                calibrated_stats["wrong_argmax_task_counter"][calibrated_argmax_task] += 1
                if calibrated_argmax_task > true_task_id:
                    calibrated_stats["wrong_to_later_task"] += 1
                    calibrated_stats["wrong_argmax_later_counter"][calibrated_argmax_task] += 1
                elif calibrated_argmax_task < true_task_id:
                    calibrated_stats["wrong_to_earlier_task"] += 1
                else:
                    calibrated_stats["wrong_to_same_task"] += 1

    return _finalize_eval_stats(baseline_stats), _finalize_eval_stats(calibrated_stats)


def main():
    cli_args = setup_parser().parse_args()
    args = load_json(cli_args.config)
    cli_dict = vars(cli_args)

    if cli_dict["batch_size"] is not None:
        args["batch_size"] = cli_dict["batch_size"]
    if cli_dict["seed"] is not None:
        args["seed"] = [cli_dict["seed"]]
    args["note"] = cli_dict["note"]

    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu")

    eval_seed = args["seed"][0] if isinstance(args["seed"], list) else int(args["seed"])
    _set_random(eval_seed)
    _set_device(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s", force=True)
    print_args(args)
    logging.info("checkpoint: %s", cli_args.checkpoint)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        eval_seed,
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    model = factory.get_model(args["model_name"], args)
    task_id = _rebuild_spie_v14_from_checkpoint(model, data_manager, checkpoint)
    total_classes = int(checkpoint["total_classes"])

    task_loaders = _build_task_loaders(
        data_manager=data_manager,
        total_classes=total_classes,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
        max_calib_per_class=cli_args.max_calib_per_class,
    )
    calibration, tau_tensor, sigma_tensor = _compute_calibration_parameters(
        model=model,
        task_loaders=task_loaders,
        tau_strategy=cli_args.tau_strategy,
        tau_quantile=float(cli_args.tau_quantile),
        sigma_eps=float(cli_args.sigma_eps),
    )

    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    test_loader = _make_loader(
        test_dataset,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
        max_per_class=cli_args.max_test_per_class,
        shuffle=False,
    )
    baseline_metrics, calibrated_metrics = _evaluate_baseline_and_calibrated(
        model=model,
        loader=test_loader,
        tau_tensor=tau_tensor,
        sigma_tensor=sigma_tensor,
    )

    results = {
        "checkpoint": cli_args.checkpoint,
        "task_id": int(task_id),
        "total_classes": int(total_classes),
        "num_experts": len(model._active_expert_ids()),
        "tau_strategy": cli_args.tau_strategy,
        "tau_quantile": float(cli_args.tau_quantile),
        "sigma_eps": float(cli_args.sigma_eps),
        "max_calib_per_class": cli_args.max_calib_per_class,
        "max_test_per_class": cli_args.max_test_per_class,
        "calibration": calibration,
        "baseline": baseline_metrics,
        "calibrated": calibrated_metrics,
        "delta": {
            "top1": _safe_round(
                (calibrated_metrics["top1"] or 0.0) - (baseline_metrics["top1"] or 0.0),
                digits=4,
            ),
            "old_top1": _safe_round(
                (calibrated_metrics["old_top1"] or 0.0) - (baseline_metrics["old_top1"] or 0.0),
                digits=4,
            )
            if baseline_metrics["old_top1"] is not None and calibrated_metrics["old_top1"] is not None
            else None,
            "new_top1": _safe_round(
                (calibrated_metrics["new_top1"] or 0.0) - (baseline_metrics["new_top1"] or 0.0),
                digits=4,
            )
            if baseline_metrics["new_top1"] is not None and calibrated_metrics["new_top1"] is not None
            else None,
            "wrong_to_later_task_rate": _safe_round(
                (calibrated_metrics["wrong_to_later_task_rate"] or 0.0)
                - (baseline_metrics["wrong_to_later_task_rate"] or 0.0),
                digits=4,
            )
            if baseline_metrics["wrong_to_later_task_rate"] is not None
            and calibrated_metrics["wrong_to_later_task_rate"] is not None
            else None,
        },
    }

    logging.info(
        "baseline top1=%s old=%s new=%s | calibrated top1=%s old=%s new=%s",
        baseline_metrics["top1"],
        baseline_metrics["old_top1"],
        baseline_metrics["new_top1"],
        calibrated_metrics["top1"],
        calibrated_metrics["old_top1"],
        calibrated_metrics["new_top1"],
    )
    logging.info(
        "wrong_to_later_task_rate baseline=%s calibrated=%s",
        baseline_metrics["wrong_to_later_task_rate"],
        calibrated_metrics["wrong_to_later_task_rate"],
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if cli_args.output:
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
