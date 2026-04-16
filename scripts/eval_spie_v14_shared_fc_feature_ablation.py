import argparse
import json
import logging
import os
import sys
from collections import defaultdict

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
        description="Pure shared-FC feature ablation for SPiE v14 checkpoints."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=None,
        help="Optional cap for final test samples per class.",
    )
    parser.add_argument("--note", type=str, default="shared-fc-feature-ablation", help="Optional logging note.")
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


def _safe_round(value, digits=4):
    return round(float(value), digits)


def _init_metric():
    return {"top1_correct": 0, "top5_correct": 0, "total": 0}


def _update_metric(metric, logits, targets):
    top1 = torch.argmax(logits, dim=1)
    top5_k = min(5, logits.shape[1])
    top5 = torch.topk(logits, k=top5_k, dim=1, largest=True, sorted=True).indices

    metric["total"] += int(targets.shape[0])
    metric["top1_correct"] += int(top1.eq(targets).sum().item())
    metric["top5_correct"] += int(top5.eq(targets.unsqueeze(1)).any(dim=1).sum().item())


def _finalize_metric(name, metric, shared_metric=None):
    total = max(int(metric["total"]), 1)
    payload = {
        "name": name,
        "top1": _safe_round(metric["top1_correct"] * 100.0 / total),
        "top5": _safe_round(metric["top5_correct"] * 100.0 / total),
        "num_samples": int(metric["total"]),
    }
    if shared_metric is not None:
        shared_total = max(int(shared_metric["total"]), 1)
        shared_top1 = shared_metric["top1_correct"] * 100.0 / shared_total
        shared_top5 = shared_metric["top5_correct"] * 100.0 / shared_total
        payload["delta_vs_shared"] = {
            "top1": _safe_round(payload["top1"] - shared_top1),
            "top5": _safe_round(payload["top5"] - shared_top5),
        }
    return payload


@torch.no_grad()
def evaluate_shared_fc_feature_ablation(model, loader):
    active_expert_ids = model._active_expert_ids()
    num_experts = len(active_expert_ids)
    backbone = model._backbone_module()
    classifier = model._network.fc_shared_cls
    total_classes = model._total_classes

    metrics = {"shared_feature": _init_metric()}
    for expert_id in active_expert_ids:
        metrics[f"expert_{expert_id}"] = _init_metric()
        metrics[f"shared_plus_expert_{expert_id}"] = _init_metric()
    if num_experts > 0:
        metrics["expert_mean_all"] = _init_metric()
        metrics["shared_plus_expert_mean_all"] = _init_metric()

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)

        multi_out = backbone.forward_multi_expert_features(inputs, active_expert_ids)
        shared_logits = classifier(multi_out["cls_features"])["logits"][:, :total_classes]
        _update_metric(metrics["shared_feature"], shared_logits, targets)

        if num_experts <= 0:
            continue

        expert_features = multi_out["expert_features"]
        flat_expert_features = expert_features.reshape(num_experts * targets.shape[0], -1)
        flat_expert_logits = classifier(flat_expert_features)["logits"][:, :total_classes]
        expert_logits = flat_expert_logits.reshape(num_experts, targets.shape[0], total_classes)

        for local_idx, expert_id in enumerate(active_expert_ids):
            cur_expert_logits = expert_logits[local_idx]
            _update_metric(metrics[f"expert_{expert_id}"], cur_expert_logits, targets)
            _update_metric(
                metrics[f"shared_plus_expert_{expert_id}"],
                0.5 * (shared_logits + cur_expert_logits),
                targets,
            )

        expert_mean_logits = expert_logits.mean(dim=0)
        _update_metric(metrics["expert_mean_all"], expert_mean_logits, targets)
        _update_metric(
            metrics["shared_plus_expert_mean_all"],
            0.5 * (shared_logits + expert_mean_logits),
            targets,
        )

    shared_metric = metrics["shared_feature"]
    expert_results = []
    for expert_id in active_expert_ids:
        payload = _finalize_metric(
            f"expert_{expert_id}",
            metrics[f"expert_{expert_id}"],
            shared_metric=shared_metric,
        )
        payload["expert_id"] = int(expert_id)
        expert_results.append(payload)
    expert_results = sorted(expert_results, key=lambda item: (-item["top1"], item["expert_id"]))

    shared_plus_expert_results = []
    for expert_id in active_expert_ids:
        payload = _finalize_metric(
            f"shared_plus_expert_{expert_id}",
            metrics[f"shared_plus_expert_{expert_id}"],
            shared_metric=shared_metric,
        )
        payload["expert_id"] = int(expert_id)
        shared_plus_expert_results.append(payload)
    shared_plus_expert_results = sorted(
        shared_plus_expert_results,
        key=lambda item: (-item["top1"], item["expert_id"]),
    )

    results = {
        "shared_feature": _finalize_metric("shared_feature", shared_metric),
        "expert_features": expert_results,
        "summary": {
            "best_expert_by_top1": expert_results[0] if expert_results else None,
            "best_shared_plus_expert_by_top1": shared_plus_expert_results[0] if shared_plus_expert_results else None,
        },
    }

    if num_experts > 0:
        results["expert_mean_all"] = _finalize_metric(
            "expert_mean_all",
            metrics["expert_mean_all"],
            shared_metric=shared_metric,
        )
        results["shared_plus_expert_mean_all"] = _finalize_metric(
            "shared_plus_expert_mean_all",
            metrics["shared_plus_expert_mean_all"],
            shared_metric=shared_metric,
        )
        results["shared_plus_expert"] = shared_plus_expert_results

    return results


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

    results = evaluate_shared_fc_feature_ablation(model=model, loader=test_loader)
    payload = {
        "checkpoint": cli_args.checkpoint,
        "task_id": int(task_id),
        "total_classes": int(total_classes),
        "num_experts": len(model._active_expert_ids()),
        "max_test_per_class": cli_args.max_test_per_class,
        "results": results,
    }

    logging.info(
        "shared top1=%s top5=%s | best expert=%s top1=%s | expert-mean top1=%s",
        results["shared_feature"]["top1"],
        results["shared_feature"]["top5"],
        results["summary"]["best_expert_by_top1"]["expert_id"] if results["summary"]["best_expert_by_top1"] else None,
        results["summary"]["best_expert_by_top1"]["top1"] if results["summary"]["best_expert_by_top1"] else None,
        results.get("expert_mean_all", {}).get("top1"),
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if cli_args.output:
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
