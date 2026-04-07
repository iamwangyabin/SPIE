import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from expert_response_analysis import (
    build_test_loader,
    find_latest_task_id,
    iter_config_seeds,
    load_json,
    load_task_checkpoint,
    resolve_checkpoint_dir,
    resolve_task_id,
    set_device,
    set_random,
)
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import accuracy


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SPiE v2 ablation evaluator: at test time, each expert feature is "
            "classified only by the classifier head trained for that same expert."
        )
    )
    parser.add_argument("--config", type=str, required=True, help="Json config file used for training.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory containing task_0.pkl, task_1.pkl, ... . Overrides checkpoint_dir in the config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional direct checkpoint path. If set, evaluates only that checkpoint.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="all",
        help="Task checkpoint to evaluate: all, latest, final, last, or an integer task id. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Evaluate only this seed. Defaults to all config seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for the test loader.")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply cosine scores by args['scale'].",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save the summary JSON.")
    return parser


def task_offsets(data_manager: DataManager, task_id: int) -> List[int]:
    offsets = []
    class_offset = 0
    for mapped_task_id in range(task_id + 1):
        offsets.append(class_offset)
        class_offset += data_manager.get_task_size(mapped_task_id)
    return offsets


def evaluate_predictions(model, y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    grouped = accuracy(
        y_pred[:, 0],
        y_true,
        model._known_classes,
        model.args["init_cls"],
        model.args["increment"],
    )
    return {
        "grouped": grouped,
        "top1": grouped["total"],
        "top{}".format(model.topk): np.around(
            (y_pred.T == np.tile(y_true, (model.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        ).item(),
    }


def own_head_logits(fc: nn.Module, features: torch.Tensor, expert_id: int) -> torch.Tensor:
    if not hasattr(fc, "heads"):
        raise TypeError("This ablation expects a TunaLinear-style classifier with per-task heads.")
    if expert_id >= len(fc.heads):
        raise ValueError(f"Missing classifier head {expert_id}; classifier has {len(fc.heads)} head(s).")

    head = fc.heads[expert_id]
    modules = list(head.children()) if isinstance(head, nn.Sequential) else [head]
    if not modules or not isinstance(modules[-1], nn.Linear):
        raise TypeError(f"Classifier head {expert_id} is not a Sequential ending in nn.Linear.")

    fc_inp = features
    for module in modules[:-1]:
        fc_inp = module(fc_inp)

    linear = modules[-1]
    return F.linear(F.normalize(fc_inp, p=2, dim=1), F.normalize(linear.weight, p=2, dim=1))


@torch.no_grad()
def predict_with_own_classifiers(
    model,
    inputs: torch.Tensor,
    active_experts: Sequence[int],
    offsets: Sequence[int],
    apply_scale: bool,
) -> torch.Tensor:
    network = model._network
    if not hasattr(network, "backbone") or not hasattr(network, "fc"):
        raise TypeError("This ablation is only for SPiE v2 / TunaMax-style models with backbone and fc.")

    logits_per_expert = []
    global_class_lookup = []
    for expert_id in active_experts:
        features = network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
        logits = own_head_logits(network.fc, features, expert_id)
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        logits_per_expert.append(logits)
        global_class_lookup.extend(range(offsets[expert_id], offsets[expert_id] + logits.shape[1]))

    logits = torch.cat(logits_per_expert, dim=1)
    topk = min(model.topk, logits.shape[1])
    topk_concat = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1]
    global_class_lookup = torch.tensor(global_class_lookup, device=logits.device, dtype=torch.long)
    predicts = global_class_lookup[topk_concat]

    if topk < model.topk:
        pad = torch.full(
            (predicts.shape[0], model.topk - topk),
            -1,
            device=predicts.device,
            dtype=predicts.dtype,
        )
        predicts = torch.cat([predicts, pad], dim=1)

    return predicts


def eval_own_classifier_ablation(model, loader, offsets: Sequence[int], apply_scale: bool) -> Dict:
    model._network.eval()
    y_pred, y_true = [], []
    active_experts = list(range(model._cur_task + 1))

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        predicts = predict_with_own_classifiers(
            model=model,
            inputs=inputs,
            active_experts=active_experts,
            offsets=offsets,
            apply_scale=apply_scale,
        )
        y_pred.append(predicts.cpu().numpy())
        y_true.append(targets.numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return evaluate_predictions(model, y_pred, y_true)


def resolve_eval_task_ids(task_id_arg: str, checkpoint_dir: Path, data_manager: DataManager) -> List[int]:
    value = str(task_id_arg).strip().lower()
    if value == "all":
        latest_task_id = find_latest_task_id(checkpoint_dir)
        return list(range(latest_task_id + 1))
    return [resolve_task_id(task_id_arg, checkpoint_dir, data_manager)]


def expected_task_for_direct_checkpoint(task_id_arg: str, data_manager: DataManager) -> Optional[int]:
    value = str(task_id_arg).strip().lower()
    if value in {"all", "latest"}:
        return None
    if value in {"final", "last"}:
        return data_manager.nb_tasks - 1
    return int(value)


def run_one_seed(config: Dict, cli_args: argparse.Namespace, seed: int) -> Dict:
    args = copy.deepcopy(config)
    args["seed"] = seed
    if cli_args.checkpoint_dir:
        args["checkpoint_dir"] = cli_args.checkpoint_dir
    if cli_args.batch_size is not None:
        args["batch_size"] = cli_args.batch_size

    set_random(args["seed"])
    set_device(args)

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

    model = factory.get_model(args["model_name"], args)
    seed_results = []

    if cli_args.checkpoint:
        checkpoint_path = Path(cli_args.checkpoint)
        expected_task_id = expected_task_for_direct_checkpoint(cli_args.task_id, data_manager)
        checkpoints = [(checkpoint_path, expected_task_id)]
    else:
        checkpoint_dir = resolve_checkpoint_dir(args)
        checkpoints = [
            (checkpoint_dir / f"task_{task_id}.pkl", task_id)
            for task_id in resolve_eval_task_ids(cli_args.task_id, checkpoint_dir, data_manager)
        ]

    for checkpoint_path, expected_task_id in checkpoints:
        logging.info("Evaluating seed=%s checkpoint=%s", seed, checkpoint_path)
        checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id)
        loader = build_test_loader(
            data_manager=data_manager,
            total_classes=model._total_classes,
            batch_size=args["batch_size"],
            num_workers=cli_args.num_workers,
        )
        offsets = task_offsets(data_manager, model._cur_task)
        own_classifier_cnn = eval_own_classifier_ablation(
            model=model,
            loader=loader,
            offsets=offsets,
            apply_scale=not cli_args.no_scale,
        )
        logging.info("Task %s own-classifier top1: %.2f", model._cur_task, own_classifier_cnn["top1"])
        seed_results.append(
            {
                "task_id": model._cur_task,
                "known_classes": model._known_classes,
                "total_classes": model._total_classes,
                "checkpoint": str(checkpoint_path),
                "own_classifier_cnn": own_classifier_cnn,
                "checkpoint_full_cnn": checkpoint.get("cnn_accy"),
                "applied_scale": not cli_args.no_scale,
                "evaluation_rule": "expert_i_features_are_classified_only_by_fc.heads[i]",
            }
        )

    return {
        "seed": seed,
        "model_name": args["model_name"],
        "dataset": args["dataset"],
        "results": seed_results,
    }


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [spie_v2_own_classifier_ablation] %(message)s")
    config = load_json(cli_args.config)

    seeds = iter_config_seeds(config, cli_args.seed)
    if cli_args.checkpoint and len(seeds) > 1 and cli_args.seed is None:
        logging.warning("--checkpoint was set with multiple config seeds; evaluating the first seed only.")
        seeds = seeds[:1]

    results = [run_one_seed(config, cli_args, seed=seed) for seed in seeds]
    output = results[0] if len(results) == 1 else results

    print(json.dumps(output, indent=2, allow_nan=True))
    if cli_args.output:
        output_path = Path(cli_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(output, f, indent=2, allow_nan=True)
        logging.info("Saved summary to %s", output_path)


if __name__ == "__main__":
    main()
