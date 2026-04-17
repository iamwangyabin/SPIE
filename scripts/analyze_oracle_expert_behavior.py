import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from main import load_json
from trainer import _set_device, _set_random, print_args
from utils import factory
from utils.data_manager import DataManager


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Analyze oracle expert behavior from a SPiE checkpoint."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument("--note", type=str, default="oracle-expert-behavior-analysis", help="Optional logging note.")
    return parser


def _materialize_v14_detector_buffers_from_checkpoint(model, state_dict):
    if not hasattr(model._network, "task_ood_detectors"):
        return

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


def _restore_v16_energy_stats_from_checkpoint(model, state_dict):
    if not hasattr(model._network, "expert_energy_mean_in"):
        return

    mean_tensor = state_dict.get("expert_energy_mean_in")
    std_tensor = state_dict.get("expert_energy_std_in")
    if mean_tensor is not None:
        model._network.expert_energy_mean_in = mean_tensor.detach().clone().to(
            device=model._device, dtype=torch.float32
        )
    if std_tensor is not None:
        model._network.expert_energy_std_in = std_tensor.detach().clone().to(
            device=model._device, dtype=torch.float32
        )


def _rebuild_model_from_checkpoint(model, data_manager, checkpoint):
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
        if hasattr(model._network, "append_expert_head"):
            model._network.append_expert_head(task_size)
        model.task_class_ranges.append((start, end))
        backbone.reset_task_modules()
        backbone.adapter_update()
        start = end

    state_dict = checkpoint["model_state_dict"]
    _materialize_v14_detector_buffers_from_checkpoint(model, state_dict)
    _restore_v16_energy_stats_from_checkpoint(model, state_dict)

    missing, unexpected = model._network.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint structure mismatch. Missing={missing}, unexpected={unexpected}")

    model._network.to(model._device)
    model._network.eval()
    return task_id


def _extract_inputs_targets(batch):
    if len(batch) != 3:
        raise ValueError(f"Expected a 3-item batch, got {len(batch)} items.")
    _, inputs, targets = batch
    return inputs, targets


@torch.no_grad()
def analyze_oracle_expert_behavior(learner, test_loader, save_path=None):
    """
    Oracle test:
    - each sample directly uses the expert of its ground-truth task
    - report overall oracle acc
    - report per-task oracle acc
    - report confidence / margin / entropy
    """

    if not hasattr(learner, "_collect_expert_logits"):
        raise AttributeError("Learner does not implement `_collect_expert_logits`.")
    if not hasattr(learner, "_class_to_task_id"):
        raise AttributeError("Learner does not implement `_class_to_task_id`.")
    if not hasattr(learner, "task_class_ranges"):
        raise AttributeError("Learner does not expose `task_class_ranges`.")

    learner._network.eval()

    total = 0
    correct = 0
    top3_correct = 0
    top5_correct = 0

    conf_sum = 0.0
    margin_sum = 0.0
    entropy_sum = 0.0
    margin_count = 0

    conf_correct_sum = 0.0
    conf_wrong_sum = 0.0
    margin_correct_sum = 0.0
    margin_wrong_sum = 0.0
    n_correct = 0
    n_wrong = 0
    margin_correct_count = 0
    margin_wrong_count = 0

    per_task = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "top3_correct": 0,
            "top5_correct": 0,
            "conf_sum": 0.0,
            "margin_sum": 0.0,
            "margin_count": 0,
            "entropy_sum": 0.0,
        }
    )

    for batch in test_loader:
        inputs, targets = _extract_inputs_targets(batch)
        inputs = inputs.to(learner._device)
        targets = targets.to(learner._device)

        gt_tasks = [learner._class_to_task_id(int(y.item())) for y in targets]
        unique_gt_tasks = sorted(set(gt_tasks))
        expert_logits_map = learner._collect_expert_logits(inputs, unique_gt_tasks)

        for i in range(targets.size(0)):
            y = int(targets[i].item())
            gt_task = gt_tasks[i]
            start_idx, _ = learner.task_class_ranges[gt_task]

            local_logits = expert_logits_map[gt_task][i]
            local_probs = F.softmax(local_logits, dim=0)

            local_topk = min(5, local_logits.numel())
            local_order = torch.argsort(local_logits, descending=True)
            pred_local = int(local_order[0].item())
            pred_global = start_idx + pred_local
            gt_local = y - start_idx

            total += 1
            per_task[gt_task]["n"] += 1

            is_correct = pred_global == y
            if is_correct:
                correct += 1
                per_task[gt_task]["correct"] += 1
            if gt_local in local_order[: min(3, local_logits.numel())]:
                top3_correct += 1
                per_task[gt_task]["top3_correct"] += 1
            if gt_local in local_order[:local_topk]:
                top5_correct += 1
                per_task[gt_task]["top5_correct"] += 1

            max_prob = float(local_probs.max().item())
            entropy = float((-(local_probs * torch.log(local_probs.clamp_min(1e-12))).sum()).item())

            top2_vals = torch.topk(local_logits, k=min(2, local_logits.numel()), largest=True).values
            if top2_vals.numel() >= 2:
                margin = float((top2_vals[0] - top2_vals[1]).item())
            else:
                margin = float("nan")

            conf_sum += max_prob
            entropy_sum += entropy
            per_task[gt_task]["conf_sum"] += max_prob
            per_task[gt_task]["entropy_sum"] += entropy

            if not math.isnan(margin):
                margin_sum += margin
                margin_count += 1
                per_task[gt_task]["margin_sum"] += margin
                per_task[gt_task]["margin_count"] += 1

            if is_correct:
                n_correct += 1
                conf_correct_sum += max_prob
                if not math.isnan(margin):
                    margin_correct_sum += margin
                    margin_correct_count += 1
            else:
                n_wrong += 1
                conf_wrong_sum += max_prob
                if not math.isnan(margin):
                    margin_wrong_sum += margin
                    margin_wrong_count += 1

    results = {
        "oracle_expert_top1": 100.0 * correct / max(total, 1),
        "oracle_expert_top3": 100.0 * top3_correct / max(total, 1),
        "oracle_expert_top5": 100.0 * top5_correct / max(total, 1),
        "avg_max_prob": conf_sum / max(total, 1),
        "avg_margin": margin_sum / max(margin_count, 1),
        "avg_entropy": entropy_sum / max(total, 1),
        "avg_max_prob_when_correct": conf_correct_sum / max(n_correct, 1),
        "avg_max_prob_when_wrong": conf_wrong_sum / max(n_wrong, 1),
        "avg_margin_when_correct": margin_correct_sum / max(margin_correct_count, 1),
        "avg_margin_when_wrong": margin_wrong_sum / max(margin_wrong_count, 1),
        "per_task": {},
    }

    for task_id in sorted(per_task.keys()):
        stat = per_task[task_id]
        n = stat["n"]
        results["per_task"][task_id] = {
            "n": n,
            "top1": 100.0 * stat["correct"] / max(n, 1),
            "top3": 100.0 * stat["top3_correct"] / max(n, 1),
            "top5": 100.0 * stat["top5_correct"] / max(n, 1),
            "avg_max_prob": stat["conf_sum"] / max(n, 1),
            "avg_margin": stat["margin_sum"] / max(stat["margin_count"], 1),
            "avg_entropy": stat["entropy_sum"] / max(n, 1),
        }

    print("\n===== oracle expert behavior =====")
    print(f"oracle expert top1: {results['oracle_expert_top1']:.2f}")
    print(f"oracle expert top3: {results['oracle_expert_top3']:.2f}")
    print(f"oracle expert top5: {results['oracle_expert_top5']:.2f}")
    print(f"avg max prob: {results['avg_max_prob']:.4f}")
    print(f"avg margin: {results['avg_margin']:.4f}")
    print(f"avg entropy: {results['avg_entropy']:.4f}")
    print(f"avg max prob when correct: {results['avg_max_prob_when_correct']:.4f}")
    print(f"avg max prob when wrong:   {results['avg_max_prob_when_wrong']:.4f}")
    print(f"avg margin when correct:   {results['avg_margin_when_correct']:.4f}")
    print(f"avg margin when wrong:     {results['avg_margin_when_wrong']:.4f}")

    print("\n----- per-task -----")
    for task_id, stat in results["per_task"].items():
        print(
            f"task {task_id:2d} | n={stat['n']:4d} | "
            f"top1={stat['top1']:.2f} | top3={stat['top3']:.2f} | top5={stat['top5']:.2f} | "
            f"maxprob={stat['avg_max_prob']:.4f} | margin={stat['avg_margin']:.4f} | entropy={stat['avg_entropy']:.4f}"
        )

    if save_path is not None:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved oracle expert analysis to: {save_path}")

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

    learner = factory.get_model(args["model_name"], args)
    task_id = _rebuild_model_from_checkpoint(learner, data_manager, checkpoint)

    total_classes = int(checkpoint["total_classes"])
    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=cli_args.num_workers,
    )

    results = analyze_oracle_expert_behavior(learner=learner, test_loader=test_loader)
    results.update(
        {
            "checkpoint": cli_args.checkpoint,
            "task_id": task_id,
            "model_name": args["model_name"],
            "total_classes": total_classes,
        }
    )

    if cli_args.output:
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info("Saved analysis JSON to %s", cli_args.output)


if __name__ == "__main__":
    main()
