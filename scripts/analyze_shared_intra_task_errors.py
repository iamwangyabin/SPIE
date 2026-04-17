import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
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
        description="Analyze shared-only same-task mistakes from a SPiE checkpoint."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument("--note", type=str, default="shared-intra-task-error-analysis", help="Optional logging note.")
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


def _safe_pct(numerator, denominator):
    return 100.0 * float(numerator) / max(int(denominator), 1)


@torch.no_grad()
def analyze_shared_intra_task_errors(learner, test_loader):
    """
    Analyze shared-only error sources:
    1) among shared top1 mistakes, how many are same-task mistakes
    2) for same-task mistakes, whether the GT class rank inside its GT-task slice is only 2 / 3
    3) whether global top2 both fall inside the GT task
    """
    learner._network.eval()

    stats = {
        "total": 0,
        "correct": 0,
        "wrong": 0,
        "same_task_wrong": 0,
        "cross_task_wrong": 0,
        "same_task_rank2": 0,
        "same_task_rank3": 0,
        "same_task_rank_le3": 0,
        "top2_both_in_gt_task_when_wrong": 0,
        "top2_swap_like_when_wrong": 0,
    }

    for _, inputs, targets in test_loader:
        inputs = inputs.to(learner._device)
        targets = targets.to(learner._device)

        shared_logits = learner._shared_cls_logits(inputs)
        pred_top1 = torch.argmax(shared_logits, dim=1)
        topk_k = min(2, shared_logits.shape[1])
        pred_top2 = torch.topk(shared_logits, k=topk_k, dim=1, largest=True, sorted=True).indices

        for sample_idx in range(targets.size(0)):
            y = int(targets[sample_idx].item())
            yhat = int(pred_top1[sample_idx].item())

            gt_task = learner._class_to_task_id(y)
            pred_task = learner._class_to_task_id(yhat)

            stats["total"] += 1
            if yhat == y:
                stats["correct"] += 1
                continue

            stats["wrong"] += 1

            if pred_task == gt_task:
                stats["same_task_wrong"] += 1

                start, end = learner.task_class_ranges[gt_task]
                local_logits = shared_logits[sample_idx, start:end]
                gt_local_idx = y - start
                local_order = torch.argsort(local_logits, descending=True)
                local_rank = int((local_order == gt_local_idx).nonzero(as_tuple=True)[0].item()) + 1

                if local_rank == 2:
                    stats["same_task_rank2"] += 1
                if local_rank == 3:
                    stats["same_task_rank3"] += 1
                if local_rank <= 3:
                    stats["same_task_rank_le3"] += 1
            else:
                stats["cross_task_wrong"] += 1

            top2_classes = [int(pred_top2[sample_idx, rank_idx].item()) for rank_idx in range(topk_k)]
            top2_tasks = [learner._class_to_task_id(class_idx) for class_idx in top2_classes]
            if len(top2_tasks) == 2 and top2_tasks[0] == gt_task and top2_tasks[1] == gt_task:
                stats["top2_both_in_gt_task_when_wrong"] += 1
                if top2_classes[1] == y:
                    stats["top2_swap_like_when_wrong"] += 1

    stats["shared_top1_acc"] = _safe_pct(stats["correct"], stats["total"])
    stats["same_task_wrong_rate_of_wrong"] = _safe_pct(stats["same_task_wrong"], stats["wrong"])
    stats["cross_task_wrong_rate_of_wrong"] = _safe_pct(stats["cross_task_wrong"], stats["wrong"])
    stats["same_task_rank2_rate_of_wrong"] = _safe_pct(stats["same_task_rank2"], stats["wrong"])
    stats["same_task_rank3_rate_of_wrong"] = _safe_pct(stats["same_task_rank3"], stats["wrong"])
    stats["same_task_rank_le3_rate_of_wrong"] = _safe_pct(stats["same_task_rank_le3"], stats["wrong"])
    stats["top2_both_in_gt_task_rate_of_wrong"] = _safe_pct(stats["top2_both_in_gt_task_when_wrong"], stats["wrong"])
    stats["top2_swap_like_rate_of_wrong"] = _safe_pct(stats["top2_swap_like_when_wrong"], stats["wrong"])
    return stats


def _print_analysis(stats):
    print("\n===== shared intra-task error analysis =====")
    print(f"total samples: {stats['total']}")
    print(f"shared top1 acc: {stats['shared_top1_acc']:.2f}")
    print(f"wrong samples: {stats['wrong']}")

    print(
        f"same-task wrong: {stats['same_task_wrong']} "
        f"({stats['same_task_wrong_rate_of_wrong']:.2f}% of wrong)"
    )
    print(
        f"cross-task wrong: {stats['cross_task_wrong']} "
        f"({stats['cross_task_wrong_rate_of_wrong']:.2f}% of wrong)"
    )

    print(
        f"same-task wrong with local rank=2: {stats['same_task_rank2']} "
        f"({stats['same_task_rank2_rate_of_wrong']:.2f}% of wrong)"
    )
    print(
        f"same-task wrong with local rank=3: {stats['same_task_rank3']} "
        f"({stats['same_task_rank3_rate_of_wrong']:.2f}% of wrong)"
    )
    print(
        f"same-task wrong with local rank<=3: {stats['same_task_rank_le3']} "
        f"({stats['same_task_rank_le3_rate_of_wrong']:.2f}% of wrong)"
    )

    print(
        f"wrong but global top2 both in gt-task: {stats['top2_both_in_gt_task_when_wrong']} "
        f"({stats['top2_both_in_gt_task_rate_of_wrong']:.2f}% of wrong)"
    )
    print(
        f'wrong and "top1/top2 just swapped inside gt-task": {stats["top2_swap_like_when_wrong"]} '
        f"({stats['top2_swap_like_rate_of_wrong']:.2f}% of wrong)"
    )


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

    stats = analyze_shared_intra_task_errors(learner, test_loader)
    stats.update(
        {
            "checkpoint": cli_args.checkpoint,
            "task_id": task_id,
            "model_name": args["model_name"],
            "total_classes": total_classes,
        }
    )
    _print_analysis(stats)

    if cli_args.output:
        output_dir = os.path.dirname(os.path.abspath(cli_args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logging.info("Saved analysis JSON to %s", cli_args.output)


if __name__ == "__main__":
    main()
