import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import load_json
from trainer import _set_device, _set_random, print_args
from utils import factory
from utils.data_manager import DataManager


EPSILON = 1e-8


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate shared-only and shared+expert prototype NME for SPiE v14 checkpoints."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=None,
        help="Optional cap for prototype-building train samples per class.",
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=None,
        help="Optional cap for final test samples per class.",
    )
    parser.add_argument(
        "--disable-feature-normalization",
        action="store_true",
        help="Disable per-sample L2 normalization before prototype construction and NME.",
    )
    parser.add_argument("--note", type=str, default="prototype-nme-eval", help="Optional logging note.")
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


def _normalize_tensor(x):
    return F.normalize(x, p=2, dim=1)


@torch.no_grad()
def _collect_shared_features(model, loader, normalize_features=True):
    labels = []
    shared = []

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)

        shared_features = model._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        if normalize_features:
            shared_features = _normalize_tensor(shared_features)

        labels.append(targets.detach().cpu())
        shared.append(shared_features.detach().cpu())

    return {
        "labels": torch.cat(labels, dim=0),
        "shared": torch.cat(shared, dim=0),
    }


@torch.no_grad()
def _collect_expert_features(model, loader, expert_id, normalize_features=True):
    expert_chunks = []
    for _, inputs, _ in loader:
        inputs = inputs.to(model._device)
        expert_features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["expert_features"]
        if normalize_features:
            expert_features = _normalize_tensor(expert_features)
        expert_chunks.append(expert_features.detach().cpu())
    return torch.cat(expert_chunks, dim=0)


def _compute_class_prototypes(features, labels, total_classes, normalize_prototypes=True):
    prototypes = []
    counts = []
    for class_idx in range(total_classes):
        class_mask = labels == class_idx
        class_count = int(class_mask.sum().item())
        if class_count <= 0:
            raise RuntimeError(f"No samples found for class {class_idx} while building prototypes.")

        class_proto = features[class_mask].mean(dim=0, keepdim=True)
        if normalize_prototypes:
            class_proto = _normalize_tensor(class_proto)
        prototypes.append(class_proto.squeeze(0))
        counts.append(class_count)

    return torch.stack(prototypes, dim=0), counts


def _evaluate_nme(query_features, query_labels, prototypes):
    query_np = query_features.cpu().numpy()
    label_np = query_labels.cpu().numpy()
    proto_np = prototypes.cpu().numpy()

    dists = ((query_np[:, None, :] - proto_np[None, :, :]) ** 2).sum(axis=2)
    topk = np.argsort(dists, axis=1)[:, : min(5, proto_np.shape[0])]
    top1 = float((topk[:, 0] == label_np).mean() * 100.0)
    top5 = float((topk == label_np[:, None]).any(axis=1).mean() * 100.0)
    return {
        "top1": _safe_round(top1),
        "top5": _safe_round(top5),
        "num_samples": int(query_features.shape[0]),
    }


def _build_concat_prototypes(
    shared_train,
    expert_train,
    labels_train,
    total_classes,
    expert_visible_range,
    normalize_prototypes=True,
):
    feature_dim = int(shared_train.shape[1])
    device = shared_train.device

    shared_prototypes, shared_counts = _compute_class_prototypes(
        shared_train,
        labels_train,
        total_classes,
        normalize_prototypes=normalize_prototypes,
    )
    concat_prototypes = torch.zeros(total_classes, feature_dim * 2, device=device, dtype=shared_train.dtype)

    start, end = expert_visible_range
    seen_mask = torch.logical_and(labels_train >= start, labels_train < end)
    seen_labels = labels_train[seen_mask]
    seen_expert = expert_train[seen_mask]

    expert_proto_map = {}
    if seen_labels.numel() > 0:
        for class_idx in range(start, end):
            class_mask = seen_labels == class_idx
            class_count = int(class_mask.sum().item())
            if class_count <= 0:
                raise RuntimeError(
                    f"Expert-visible class {class_idx} has no samples for range [{start}, {end})."
                )
            class_proto = seen_expert[class_mask].mean(dim=0, keepdim=True)
            if normalize_prototypes:
                class_proto = _normalize_tensor(class_proto)
            expert_proto_map[class_idx] = class_proto.squeeze(0)

    zero_pad = torch.zeros(feature_dim, device=device, dtype=shared_train.dtype)
    for class_idx in range(total_classes):
        expert_part = expert_proto_map.get(class_idx, zero_pad)
        concat_prototypes[class_idx] = torch.cat((shared_prototypes[class_idx], expert_part), dim=0)

    if normalize_prototypes:
        concat_prototypes = _normalize_tensor(concat_prototypes)
    return concat_prototypes, shared_counts


def _build_concat_queries(shared_test, expert_test, normalize_queries=True):
    concat_queries = torch.cat((shared_test, expert_test), dim=1)
    if normalize_queries:
        concat_queries = _normalize_tensor(concat_queries)
    return concat_queries


@torch.no_grad()
def evaluate_prototype_nme(model, train_loader, test_loader, normalize_features=True):
    active_expert_ids = model._active_expert_ids()
    total_classes = model._total_classes

    train_features = _collect_shared_features(model=model, loader=train_loader, normalize_features=normalize_features)
    test_features = _collect_shared_features(model=model, loader=test_loader, normalize_features=normalize_features)

    shared_prototypes, train_counts = _compute_class_prototypes(
        train_features["shared"],
        train_features["labels"],
        total_classes,
        normalize_prototypes=normalize_features,
    )
    shared_result = _evaluate_nme(
        query_features=test_features["shared"],
        query_labels=test_features["labels"],
        prototypes=shared_prototypes,
    )

    per_expert_results = []
    for expert_id in active_expert_ids:
        expert_id = int(expert_id)
        train_expert_features = _collect_expert_features(
            model=model,
            loader=train_loader,
            expert_id=expert_id,
            normalize_features=normalize_features,
        )
        test_expert_features = _collect_expert_features(
            model=model,
            loader=test_loader,
            expert_id=expert_id,
            normalize_features=normalize_features,
        )
        concat_prototypes, _ = _build_concat_prototypes(
            shared_train=train_features["shared"],
            expert_train=train_expert_features,
            labels_train=train_features["labels"],
            total_classes=total_classes,
            expert_visible_range=model.task_class_ranges[expert_id],
            normalize_prototypes=normalize_features,
        )
        concat_queries = _build_concat_queries(
            shared_test=test_features["shared"],
            expert_test=test_expert_features,
            normalize_queries=normalize_features,
        )
        metrics = _evaluate_nme(
            query_features=concat_queries,
            query_labels=test_features["labels"],
            prototypes=concat_prototypes,
        )
        metrics["expert_id"] = expert_id
        metrics["visible_class_range"] = list(model.task_class_ranges[expert_id])
        metrics["num_visible_classes"] = int(model.task_class_ranges[expert_id][1] - model.task_class_ranges[expert_id][0])
        metrics["prototype_policy"] = "shared_for_all_classes + expert_only_for_visible_classes + zero_pad_for_others"
        per_expert_results.append(metrics)

    per_expert_results = sorted(per_expert_results, key=lambda item: (-item["top1"], item["expert_id"]))
    mean_top1 = np.mean([item["top1"] for item in per_expert_results]) if per_expert_results else float("nan")
    mean_top5 = np.mean([item["top5"] for item in per_expert_results]) if per_expert_results else float("nan")

    return {
        "shared_only_nme": {
            **shared_result,
            "num_classes": int(total_classes),
            "train_samples_per_class": [int(count) for count in train_counts],
        },
        "concat_shared_expert_nme": per_expert_results,
        "summary": {
            "best_expert_by_top1": per_expert_results[0] if per_expert_results else None,
            "average_concat_top1": _safe_round(mean_top1) if per_expert_results else None,
            "average_concat_top5": _safe_round(mean_top5) if per_expert_results else None,
        },
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
    train_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="train", mode="test")
    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    train_loader = _make_loader(
        train_dataset,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
        max_per_class=cli_args.max_train_per_class,
        shuffle=False,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
        max_per_class=cli_args.max_test_per_class,
        shuffle=False,
    )

    results = evaluate_prototype_nme(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        normalize_features=not cli_args.disable_feature_normalization,
    )
    payload = {
        "checkpoint": cli_args.checkpoint,
        "task_id": int(task_id),
        "total_classes": int(total_classes),
        "num_experts": len(model._active_expert_ids()),
        "max_train_per_class": cli_args.max_train_per_class,
        "max_test_per_class": cli_args.max_test_per_class,
        "feature_normalization": not cli_args.disable_feature_normalization,
        "results": results,
    }

    best_expert = results["summary"]["best_expert_by_top1"]
    logging.info(
        "shared-only NME top1=%s top5=%s | best concat expert=%s top1=%s top5=%s | avg concat top1=%s",
        results["shared_only_nme"]["top1"],
        results["shared_only_nme"]["top5"],
        best_expert["expert_id"] if best_expert else None,
        best_expert["top1"] if best_expert else None,
        best_expert["top5"] if best_expert else None,
        results["summary"]["average_concat_top1"],
    )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if cli_args.output:
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
