import argparse
import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import load_json
from trainer import _set_device, _set_random
from utils import factory
from utils.data_manager import DataManager

NUM_WORKERS = 8


def scan_checkpoints(ckpt_dir: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    for path in ckpt_dir.glob("task_*.pkl"):
        match = re.fullmatch(r"task_(\d+)\.pkl", path.name)
        if match:
            found[int(match.group(1))] = path
    if not found:
        raise FileNotFoundError(f"No task_*.pkl checkpoints found in: {ckpt_dir}")
    return dict(sorted(found.items(), key=lambda item: item[0]))


def build_model_for_checkpoint(args: Dict[str, Any], checkpoint: Dict[str, Any], data_manager: DataManager):
    model = factory.get_model(args["model_name"], args)
    target_task = int(checkpoint["tasks"])

    for task_id in range(target_task + 1):
        model._cur_task += 1
        model._total_classes = model._known_classes + data_manager.get_task_size(task_id)
        current_task_size = model._total_classes - model._known_classes
        model.task_class_ranges.append((model._known_classes, model._total_classes))
        model._network.update_fc(current_task_size)
        if hasattr(model._network, "append_expert_head"):
            model._network.append_expert_head(current_task_size)
        if hasattr(model, "_should_reset_task_modules") and model._should_reset_task_modules():
            model._backbone_module().reset_task_modules()
        model._known_classes = model._total_classes

    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = model._network.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint structure mismatch. Missing={missing}, unexpected={unexpected}")

    model._network.to(model._device)
    model._network.eval()
    model.data_manager = data_manager
    return model


def compute_task_class_ranges(data_manager: DataManager, num_tasks: int) -> List[Tuple[int, int]]:
    ranges = []
    start = 0
    for task_id in range(num_tasks):
        end = start + int(data_manager.get_task_size(task_id))
        ranges.append((start, end))
        start = end
    return ranges


def build_task_loader(data_manager: DataManager, start: int, end: int, split: str, batch_size: int) -> DataLoader:
    source = "test" if split == "test" else "train"
    dataset = data_manager.get_dataset(np.arange(start, end), source=source, mode="test")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)


@torch.no_grad()
def eval_expert_on_task(model, loader: DataLoader, task_id: int) -> Dict[str, Any]:
    start, _ = model.task_class_ranges[task_id]
    total = 0
    correct = 0
    top5_correct = 0
    conf_sum = 0.0
    margin_sum = 0.0
    preds: List[np.ndarray] = []

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        expert_logits_map = model._collect_expert_logits(inputs, [task_id])
        local_logits = expert_logits_map[task_id]
        local_probs = F.softmax(local_logits, dim=1)
        local_order = torch.argsort(local_logits, dim=1, descending=True)
        pred_local = local_order[:, 0]
        pred_global = pred_local + start
        gt_local = targets - start

        preds.append(pred_global.cpu().numpy())
        total += targets.size(0)
        correct += int((pred_global == targets).sum().item())

        k = min(5, local_logits.shape[1])
        topk_local = local_order[:, :k]
        top5_correct += int((topk_local == gt_local.unsqueeze(1)).any(dim=1).sum().item())

        conf_sum += float(local_probs.max(dim=1).values.sum().item())
        if local_logits.shape[1] >= 2:
            top2_vals = torch.topk(local_logits, k=2, dim=1, largest=True, sorted=True).values
            margin_sum += float((top2_vals[:, 0] - top2_vals[:, 1]).sum().item())
        else:
            margin_sum += float(local_logits[:, 0].sum().item())

    return {
        "top1": 100.0 * correct / max(total, 1),
        "top5": 100.0 * top5_correct / max(total, 1),
        "avg_max_prob": conf_sum / max(total, 1),
        "avg_margin": margin_sum / max(total, 1),
        "preds": np.concatenate(preds, axis=0),
    }


@torch.no_grad()
def eval_shared_local_on_task(model, loader: DataLoader, task_id: int) -> Dict[str, Any]:
    start, end = model.task_class_ranges[task_id]
    total = 0
    correct = 0
    top5_correct = 0
    preds: List[np.ndarray] = []

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        shared_logits = model._shared_cls_logits(inputs)
        local_logits = shared_logits[:, start:end]
        local_order = torch.argsort(local_logits, dim=1, descending=True)
        pred_local = local_order[:, 0]
        pred_global = pred_local + start
        gt_local = targets - start

        preds.append(pred_global.cpu().numpy())
        total += targets.size(0)
        correct += int((pred_global == targets).sum().item())

        k = min(5, local_logits.shape[1])
        topk_local = local_order[:, :k]
        top5_correct += int((topk_local == gt_local.unsqueeze(1)).any(dim=1).sum().item())

    return {
        "top1": 100.0 * correct / max(total, 1),
        "top5": 100.0 * top5_correct / max(total, 1),
        "preds": np.concatenate(preds, axis=0),
    }


def default_expert_patterns(task_id: int) -> List[str]:
    return [
        rf"(?:^|[._])experts[._]{task_id}(?:[._]|$)",
        rf"(?:^|[._])expert_tokens[._]{task_id}(?:[._]|$)",
        rf"(?:^|[._])expert_adapters[._]{task_id}(?:[._]|$)",
        rf"(?:^|[._])task_experts[._]{task_id}(?:[._]|$)",
        rf"(?:^|[._])expert_heads[._]{task_id}(?:[._]|$)",
        rf"(?:^|[._])local_heads[._]{task_id}(?:[._]|$)",
        rf"(?:^|[._])fc_list[._]{task_id}(?:[._]|$)",
        rf"expert.*(?:^|[._]){task_id}(?:[._]|$)",
    ]


def load_state_dict_from_ckpt(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def match_keys(sd_keys: List[str], task_id: int, patterns: List[str]) -> List[str]:
    compiled = [re.compile(p) for p in (patterns if patterns else default_expert_patterns(task_id))]
    matched = []
    for key in sd_keys:
        if any(pattern.search(key) for pattern in compiled):
            matched.append(key)
    return sorted(set(matched))


def compare_task_weights(anchor_ckpt: Path, later_ckpt: Path, task_id: int, patterns: List[str]) -> Dict[str, Any]:
    state_dict_anchor = load_state_dict_from_ckpt(anchor_ckpt)
    state_dict_later = load_state_dict_from_ckpt(later_ckpt)
    keys_anchor = match_keys(list(state_dict_anchor.keys()), task_id, patterns)
    keys_later = match_keys(list(state_dict_later.keys()), task_id, patterns)
    union_keys = sorted(set(keys_anchor) | set(keys_later))

    changed = []
    missing_anchor = [key for key in union_keys if key not in state_dict_anchor]
    missing_later = [key for key in union_keys if key not in state_dict_later]

    for key in union_keys:
        if key not in state_dict_anchor or key not in state_dict_later:
            continue
        tensor_anchor, tensor_later = state_dict_anchor[key], state_dict_later[key]
        if not torch.is_tensor(tensor_anchor) or not torch.is_tensor(tensor_later):
            continue
        if tensor_anchor.shape != tensor_later.shape:
            changed.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "shape_anchor": list(tensor_anchor.shape),
                    "shape_later": list(tensor_later.shape),
                }
            )
            continue
        if tensor_sha256(tensor_anchor) != tensor_sha256(tensor_later):
            changed.append(
                {
                    "key": key,
                    "max_abs_diff": float((tensor_anchor - tensor_later).abs().max().item()),
                    "mean_abs_diff": float((tensor_anchor - tensor_later).abs().mean().item()),
                }
            )

    return {
        "num_matched_anchor": len(keys_anchor),
        "num_matched_later": len(keys_later),
        "num_union": len(union_keys),
        "missing_in_anchor": missing_anchor,
        "missing_in_later": missing_later,
        "num_changed": len(changed),
        "all_identical": len(changed) == 0 and not missing_anchor and not missing_later,
        "changed": changed[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan task checkpoints and test whether each expert drifts after its own task."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing task_0.pkl ... task_N.pkl.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--output", type=str, default="", help="Optional JSON report path.")
    parser.add_argument("--compare_weights", action="store_true")
    parser.add_argument(
        "--expert_key_regex",
        action="append",
        default=[],
        help="Optional regex for expert-weight matching; repeatable. Defaults to broad expert patterns.",
    )
    parser.add_argument(
        "--drift_warn_drop",
        type=float,
        default=3.0,
        help="Warn if expert top1 drops more than this many points from its anchor checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    ckpt_dir = Path(cli.ckpt_dir)
    checkpoints = scan_checkpoints(ckpt_dir)

    args = copy.deepcopy(load_json(cli.config))
    _set_device(args)
    seed = args["seed"][0] if isinstance(args.get("seed", 1), list) else int(args.get("seed", 1))
    _set_random(seed)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        seed,
        args["init_cls"],
        args["increment"],
        args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    max_task = max(checkpoints.keys())
    task_ranges = compute_task_class_ranges(data_manager, max_task + 1)
    batch_size = int(args.get("batch_size", 32))
    task_loaders = {
        task_id: build_task_loader(data_manager, start, end, cli.split, batch_size)
        for task_id, (start, end) in enumerate(task_ranges)
    }

    report: Dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir),
        "split": cli.split,
        "tasks": {},
        "warnings": [],
    }

    baseline_preds_expert: Dict[int, np.ndarray] = {}
    baseline_preds_shared: Dict[int, np.ndarray] = {}
    baseline_metrics: Dict[int, Dict[str, Any]] = {}

    for ckpt_id, ckpt_path in checkpoints.items():
        print(f"\n===== evaluating checkpoint task_{ckpt_id}.pkl =====")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model = build_model_for_checkpoint(args, checkpoint, data_manager)

        if not hasattr(model, "_collect_expert_logits"):
            raise AttributeError(f"Model '{args['model_name']}' does not implement `_collect_expert_logits`.")
        if not hasattr(model, "_shared_cls_logits"):
            raise AttributeError(f"Model '{args['model_name']}' does not implement `_shared_cls_logits`.")

        for task_id in range(ckpt_id + 1):
            loader = task_loaders[task_id]
            expert_metrics = eval_expert_on_task(model, loader, task_id)
            shared_metrics = eval_shared_local_on_task(model, loader, task_id)

            if task_id not in report["tasks"]:
                report["tasks"][task_id] = {"baseline_ckpt": task_id, "evaluations": {}}

            entry: Dict[str, Any] = {
                "expert_top1": expert_metrics["top1"],
                "expert_top5": expert_metrics["top5"],
                "expert_avg_max_prob": expert_metrics["avg_max_prob"],
                "expert_avg_margin": expert_metrics["avg_margin"],
                "shared_local_top1": shared_metrics["top1"],
                "shared_local_top5": shared_metrics["top5"],
            }

            if ckpt_id == task_id:
                baseline_preds_expert[task_id] = expert_metrics["preds"]
                baseline_preds_shared[task_id] = shared_metrics["preds"]
                baseline_metrics[task_id] = {
                    "expert_top1": expert_metrics["top1"],
                    "shared_local_top1": shared_metrics["top1"],
                }
                entry["delta_expert_top1_vs_anchor"] = 0.0
                entry["delta_shared_local_top1_vs_anchor"] = 0.0
                entry["expert_pred_agreement_vs_anchor"] = 100.0
                entry["shared_local_pred_agreement_vs_anchor"] = 100.0
            else:
                entry["delta_expert_top1_vs_anchor"] = expert_metrics["top1"] - baseline_metrics[task_id]["expert_top1"]
                entry["delta_shared_local_top1_vs_anchor"] = (
                    shared_metrics["top1"] - baseline_metrics[task_id]["shared_local_top1"]
                )
                entry["expert_pred_agreement_vs_anchor"] = 100.0 * float(
                    (expert_metrics["preds"] == baseline_preds_expert[task_id]).mean()
                )
                entry["shared_local_pred_agreement_vs_anchor"] = 100.0 * float(
                    (shared_metrics["preds"] == baseline_preds_shared[task_id]).mean()
                )

                if entry["delta_expert_top1_vs_anchor"] <= -abs(cli.drift_warn_drop):
                    report["warnings"].append(
                        f"task {task_id}: expert top1 dropped {entry['delta_expert_top1_vs_anchor']:.2f} points "
                        f"by checkpoint {ckpt_id}."
                    )

            if cli.compare_weights:
                patterns = cli.expert_key_regex if cli.expert_key_regex else default_expert_patterns(task_id)
                entry["weight_compare_vs_anchor"] = compare_task_weights(checkpoints[task_id], ckpt_path, task_id, patterns)

            report["tasks"][task_id]["evaluations"][ckpt_id] = entry

            print(
                f"task {task_id:2d} @ ckpt {ckpt_id:2d} | "
                f"expert top1={entry['expert_top1']:.2f} "
                f"(delta {entry['delta_expert_top1_vs_anchor']:+.2f}) | "
                f"shared-local top1={entry['shared_local_top1']:.2f} "
                f"(delta {entry['delta_shared_local_top1_vs_anchor']:+.2f}) | "
                f"expert agreement={entry['expert_pred_agreement_vs_anchor']:.2f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n===== summary of suspicious drift =====")
    if report["warnings"]:
        for warning in report["warnings"]:
            print("-", warning)
    else:
        print("No expert top1 drop exceeded threshold.")

    output_path = Path(cli.output) if cli.output else ckpt_dir / f"expert_drift_scan_{cli.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
