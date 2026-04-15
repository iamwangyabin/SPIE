import argparse
import json
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from main import load_json
from trainer import _set_device, _set_random, print_args
from utils import factory
from utils.data_manager import DataManager


def setup_parser():
    parser = argparse.ArgumentParser(description="Evaluate per-expert OOD blocking for SPiE v14 checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_4.pkl.")
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=None,
        help="Optional cap for final test samples per class.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--note", type=str, default="expert-ood-eval", help="Optional logging note.")
    return parser


def _subset_dataset_per_class(dataset, max_per_class):
    if max_per_class is None:
        return dataset

    label_to_indices = defaultdict(list)
    labels = np.asarray(dataset.labels)
    for idx, label in enumerate(labels):
        if len(label_to_indices[int(label)]) < max_per_class:
            label_to_indices[int(label)].append(idx)

    subset_indices = []
    for label in sorted(label_to_indices.keys()):
        subset_indices.extend(label_to_indices[label])
    return Subset(dataset, subset_indices)


def _make_loader(dataset, batch_size, num_workers, max_per_class=None, shuffle=False):
    dataset = _subset_dataset_per_class(dataset, max_per_class)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
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

    missing, unexpected = model._network.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint structure mismatch. Missing={missing}, unexpected={unexpected}")

    model._network.to(model._device)
    model._network.eval()
    return task_id


@torch.no_grad()
def _collect_expert_scores_and_labels(model, loader, expert_id):
    detector = model._network.get_task_ood_detector(expert_id)
    task_start, task_end = model.task_class_ranges[expert_id]
    score_chunks = []
    label_chunks = []

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["expert_features"]
        scores = detector.score_from_stats(detector.compute_stats(features))
        score_chunks.append(scores.detach().cpu())
        labels = torch.logical_and(targets >= task_start, targets < task_end).to(dtype=torch.int64)
        label_chunks.append(labels.cpu())

    if not score_chunks:
        return torch.zeros(0, dtype=torch.float32), torch.zeros(0, dtype=torch.int64)
    return torch.cat(score_chunks, dim=0), torch.cat(label_chunks, dim=0)


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


def _compute_auroc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_count = int(y_true.sum())
    neg_count = int((1 - y_true).sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fpr = np.r_[0.0, fps / neg_count, 1.0]
    tpr = np.r_[0.0, tps / pos_count, 1.0]
    return float(np.trapz(tpr, fpr))


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
    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    test_loader = _make_loader(
        test_dataset,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
        max_per_class=cli_args.max_test_per_class,
        shuffle=False,
    )

    results = {
        "checkpoint": cli_args.checkpoint,
        "task_id": task_id,
        "total_classes": total_classes,
        "max_test_per_class": cli_args.max_test_per_class,
        "experts": [],
    }

    for expert_id, (start, end) in enumerate(model.task_class_ranges):
        scores, labels = _collect_expert_scores_and_labels(model, test_loader, expert_id)
        scores_np = scores.numpy()
        labels_np = labels.numpy()
        auroc = _compute_auroc(labels_np, scores_np)
        fpr95, threshold95 = _compute_fpr95(labels_np, scores_np, target_tpr=0.95)
        id_mask = labels_np == 1
        ood_mask = labels_np == 0

        expert_metrics = {
            "expert_id": expert_id,
            "task_range": [start, end],
            "num_id": int(id_mask.sum()),
            "num_ood": int(ood_mask.sum()),
            "auroc": round(auroc, 6) if not np.isnan(auroc) else None,
            "fpr95": round(fpr95, 6) if not np.isnan(fpr95) else None,
            "threshold_at_tpr95": round(threshold95, 6) if not np.isnan(threshold95) else None,
            "id_score_mean": round(float(scores_np[id_mask].mean()), 6) if id_mask.any() else None,
            "ood_score_mean": round(float(scores_np[ood_mask].mean()), 6) if ood_mask.any() else None,
        }
        results["experts"].append(expert_metrics)

        logging.info(
            "expert=%s task=%s-%s num_id=%s num_ood=%s auroc=%s fpr95=%s threshold@tpr95=%s",
            expert_id,
            start,
            end,
            expert_metrics["num_id"],
            expert_metrics["num_ood"],
            expert_metrics["auroc"],
            expert_metrics["fpr95"],
            expert_metrics["threshold_at_tpr95"],
        )

    if results["experts"]:
        valid_aurocs = [item["auroc"] for item in results["experts"] if item["auroc"] is not None]
        valid_fpr95s = [item["fpr95"] for item in results["experts"] if item["fpr95"] is not None]
        results["mean_auroc"] = round(float(np.mean(valid_aurocs)), 6) if valid_aurocs else None
        results["mean_fpr95"] = round(float(np.mean(valid_fpr95s)), 6) if valid_fpr95s else None

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if cli_args.output:
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
