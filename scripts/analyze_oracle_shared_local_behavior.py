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
        description="Analyze oracle shared-local behavior from a SPiE checkpoint."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument("--note", type=str, default="oracle-shared-local-behavior-analysis", help="Optional logging note.")
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
def analyze_oracle_shared_local_top1(learner, test_loader):
    if not hasattr(learner, "_shared_cls_logits"):
        raise AttributeError("Learner does not implement `_shared_cls_logits`.")
    if not hasattr(learner, "_class_to_task_id"):
        raise AttributeError("Learner does not implement `_class_to_task_id`.")
    if not hasattr(learner, "task_class_ranges"):
        raise AttributeError("Learner does not expose `task_class_ranges`.")

    learner._network.eval()

    total = 0
    correct = 0
    top3_correct = 0
    top5_correct = 0

    for batch in test_loader:
        inputs, targets = _extract_inputs_targets(batch)
        inputs = inputs.to(learner._device)
        targets = targets.to(learner._device)

        shared_logits = learner._shared_cls_logits(inputs)

        for i in range(targets.size(0)):
            y = int(targets[i].item())
            gt_task = learner._class_to_task_id(y)
            start, end = learner.task_class_ranges[gt_task]

            local_logits = shared_logits[i, start:end]
            local_order = torch.argsort(local_logits, descending=True)

            gt_local = y - start
            pred_local = int(local_order[0].item())

            total += 1
            if pred_local == gt_local:
                correct += 1
            if gt_local in local_order[: min(3, local_logits.numel())]:
                top3_correct += 1
            if gt_local in local_order[: min(5, local_logits.numel())]:
                top5_correct += 1

    results = {
        "oracle_shared_local_top1": 100.0 * correct / max(total, 1),
        "oracle_shared_local_top3": 100.0 * top3_correct / max(total, 1),
        "oracle_shared_local_top5": 100.0 * top5_correct / max(total, 1),
        "total": total,
        "correct": correct,
    }

    print("\n===== oracle shared-local behavior =====")
    print(f"oracle shared-local top1: {results['oracle_shared_local_top1']:.2f}")
    print(f"oracle shared-local top3: {results['oracle_shared_local_top3']:.2f}")
    print(f"oracle shared-local top5: {results['oracle_shared_local_top5']:.2f}")

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

    results = analyze_oracle_shared_local_top1(learner=learner, test_loader=test_loader)
    results.update(
        {
            "checkpoint": cli_args.checkpoint,
            "task_id": task_id,
            "model_name": args["model_name"],
            "total_classes": total_classes,
        }
    )

    if cli_args.output:
        output_dir = os.path.dirname(os.path.abspath(cli_args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(cli_args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info("Saved analysis JSON to %s", cli_args.output)


if __name__ == "__main__":
    main()
