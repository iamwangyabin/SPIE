import argparse
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
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
        "--threshold-quantile",
        type=float,
        default=0.05,
        help="Acceptance threshold quantile computed from in-task train scores.",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=None,
        help="Optional cap for threshold-estimation samples per class.",
    )
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
def _collect_expert_scores(model, loader, expert_id):
    detector = model._network.get_task_ood_detector(expert_id)
    score_chunks = []

    for _, inputs, _ in loader:
        inputs = inputs.to(model._device)
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["expert_features"]
        scores = detector.score_from_stats(detector.compute_stats(features))
        score_chunks.append(scores.detach().cpu())

    if not score_chunks:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(score_chunks, dim=0)


@torch.no_grad()
def _evaluate_single_expert(model, loader, expert_id, threshold):
    detector = model._network.get_task_ood_detector(expert_id)
    task_start, task_end = model.task_class_ranges[expert_id]

    in_total = 0
    in_accept = 0
    out_total = 0
    out_block = 0

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)

        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["expert_features"]
        scores = detector.score_from_stats(detector.compute_stats(features))
        accepted = scores >= threshold
        in_mask = torch.logical_and(targets >= task_start, targets < task_end)
        out_mask = ~in_mask

        in_total += int(in_mask.sum().item())
        in_accept += int(torch.logical_and(accepted, in_mask).sum().item())
        out_total += int(out_mask.sum().item())
        out_block += int(torch.logical_and(~accepted, out_mask).sum().item())

    total = in_total + out_total
    correct = in_accept + out_block
    return {
        "expert_id": expert_id,
        "task_range": [task_start, task_end],
        "threshold": float(threshold),
        "in_task_total": in_total,
        "in_task_accept": in_accept,
        "in_task_reject": in_total - in_accept,
        "in_task_accept_rate": round(in_accept / max(in_total, 1), 4),
        "ood_total": out_total,
        "ood_block": out_block,
        "ood_leak": out_total - out_block,
        "ood_block_rate": round(out_block / max(out_total, 1), 4),
        "binary_acc": round(correct / max(total, 1), 4),
    }


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
        "threshold_quantile": cli_args.threshold_quantile,
        "max_train_per_class": cli_args.max_train_per_class,
        "max_test_per_class": cli_args.max_test_per_class,
        "experts": [],
    }

    for expert_id, (start, end) in enumerate(model.task_class_ranges):
        train_dataset = data_manager.get_dataset(np.arange(start, end), source="train", mode="test")
        train_loader = _make_loader(
            train_dataset,
            batch_size=args["batch_size"],
            num_workers=cli_args.num_workers,
            max_per_class=cli_args.max_train_per_class,
            shuffle=False,
        )
        positive_scores = _collect_expert_scores(model, train_loader, expert_id)
        if positive_scores.numel() == 0:
            raise RuntimeError(f"Expert {expert_id} has no positive scores for threshold estimation.")

        threshold = torch.quantile(positive_scores, q=cli_args.threshold_quantile).item()
        expert_metrics = _evaluate_single_expert(model, test_loader, expert_id, threshold)
        expert_metrics["threshold_score_min"] = float(positive_scores.min().item())
        expert_metrics["threshold_score_mean"] = float(positive_scores.mean().item())
        expert_metrics["threshold_score_max"] = float(positive_scores.max().item())
        results["experts"].append(expert_metrics)

        logging.info(
            "expert=%s task=%s-%s threshold=%.4f in_accept=%.4f ood_block=%.4f binary_acc=%.4f",
            expert_id,
            start,
            end,
            expert_metrics["threshold"],
            expert_metrics["in_task_accept_rate"],
            expert_metrics["ood_block_rate"],
            expert_metrics["binary_acc"],
        )

    if results["experts"]:
        results["avg_ood_block_rate"] = round(
            float(np.mean([item["ood_block_rate"] for item in results["experts"]])),
            4,
        )
        results["avg_in_task_accept_rate"] = round(
            float(np.mean([item["in_task_accept_rate"] for item in results["experts"]])),
            4,
        )
        results["avg_binary_acc"] = round(
            float(np.mean([item["binary_acc"] for item in results["experts"]])),
            4,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if cli_args.output:
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
