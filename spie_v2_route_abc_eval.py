import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from expert_response_analysis import (
    build_task_metadata,
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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SPiE v2 route/classification decomposition: "
            "A=P(t_hat=t*), B=P(y_hat=y|t_hat=t*), C=P(y_hat=y|t_hat!=t*)."
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
        default="latest",
        help="Task checkpoint to evaluate: all, latest, final, last, or an integer task id. Default: latest.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Evaluate only this seed. Defaults to all config seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for the test loader.")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply logits by args['scale'] before route/class prediction.",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save the summary JSON.")
    return parser


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


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def weighted_term(weight: float, conditional_probability: float) -> float:
    if weight == 0.0:
        return 0.0
    return weight * conditional_probability


@torch.no_grad()
def eval_route_abc(model, loader, class_to_task: Dict[int, int], apply_scale: bool) -> Dict:
    model._network.eval()
    expert_ids = list(range(model._cur_task + 1))

    num_samples = 0
    route_correct = 0
    route_wrong = 0
    correct_given_route_correct = 0
    correct_given_route_wrong = 0
    overall_correct = 0

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)

        logits_per_expert = []
        for expert_id in expert_ids:
            features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
            logits = model._network.fc(features)["logits"][:, : model._total_classes]
            if apply_scale:
                logits = logits * float(model.args.get("scale", 1.0))
            logits_per_expert.append(logits)

        stacked_logits = torch.stack(logits_per_expert, dim=0)
        route_scores = stacked_logits.max(dim=2).values.transpose(0, 1)
        best_expert_positions = route_scores.argmax(dim=1)
        selected_logits = stacked_logits.permute(1, 0, 2)[
            torch.arange(stacked_logits.shape[1], device=stacked_logits.device),
            best_expert_positions,
        ]
        y_hat = selected_logits.argmax(dim=1)

        expert_id_tensor = torch.tensor(expert_ids, device=targets.device, dtype=torch.long)
        t_hat = expert_id_tensor[best_expert_positions]
        t_star = torch.tensor(
            [class_to_task[int(target.item())] for target in targets],
            device=targets.device,
            dtype=torch.long,
        )

        route_is_correct = t_hat == t_star
        class_is_correct = y_hat == targets

        batch_route_correct = int(route_is_correct.sum().item())
        batch_correct_given_route_correct = int((route_is_correct & class_is_correct).sum().item())
        batch_correct_given_route_wrong = int((~route_is_correct & class_is_correct).sum().item())

        num_samples += int(targets.shape[0])
        route_correct += batch_route_correct
        route_wrong += int(targets.shape[0]) - batch_route_correct
        correct_given_route_correct += batch_correct_given_route_correct
        correct_given_route_wrong += batch_correct_given_route_wrong
        overall_correct += int(class_is_correct.sum().item())

    a = safe_divide(route_correct, num_samples)
    b = safe_divide(correct_given_route_correct, route_correct)
    c = safe_divide(correct_given_route_wrong, route_wrong)
    full_acc = safe_divide(overall_correct, num_samples)
    full_acc_from_decomposition = weighted_term(a, b) + weighted_term(1.0 - a, c)

    return {
        "num_samples": num_samples,
        "counts": {
            "route_correct": route_correct,
            "route_wrong": route_wrong,
            "class_correct_given_route_correct": correct_given_route_correct,
            "class_correct_given_route_wrong": correct_given_route_wrong,
            "class_correct_total": overall_correct,
        },
        "A_task_id_accuracy": a,
        "B_acc_given_task_id_correct": b,
        "C_acc_given_task_id_wrong": c,
        "full_acc_direct": full_acc,
        "full_acc_from_decomposition": full_acc_from_decomposition,
        "A_task_id_accuracy_percent": a * 100.0,
        "B_acc_given_task_id_correct_percent": b * 100.0,
        "C_acc_given_task_id_wrong_percent": c * 100.0,
        "full_acc_direct_percent": full_acc * 100.0,
        "full_acc_from_decomposition_percent": full_acc_from_decomposition * 100.0,
    }


def build_data_manager(args: Dict) -> DataManager:
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
    return data_manager


def checkpoint_plan(
    args: Dict,
    cli_args: argparse.Namespace,
    data_manager: DataManager,
) -> Tuple[List[Tuple[Path, Optional[int]]], Optional[Path]]:
    if cli_args.checkpoint:
        checkpoint_path = Path(cli_args.checkpoint)
        expected_task_id = expected_task_for_direct_checkpoint(cli_args.task_id, data_manager)
        return [(checkpoint_path, expected_task_id)], None

    checkpoint_dir = resolve_checkpoint_dir(args)
    task_ids = resolve_eval_task_ids(cli_args.task_id, checkpoint_dir, data_manager)
    return [(checkpoint_dir / f"task_{task_id}.pkl", task_id) for task_id in task_ids], checkpoint_dir


def run_one_seed(config: Dict, cli_args: argparse.Namespace, seed: int) -> Dict:
    args = copy.deepcopy(config)
    args["seed"] = seed
    if cli_args.checkpoint_dir:
        args["checkpoint_dir"] = cli_args.checkpoint_dir
    if cli_args.batch_size is not None:
        args["batch_size"] = cli_args.batch_size

    set_random(args["seed"])
    set_device(args)
    data_manager = build_data_manager(args)
    model = factory.get_model(args["model_name"], args)
    checkpoints, _ = checkpoint_plan(args, cli_args, data_manager)

    seed_results = []
    for checkpoint_path, expected_task_id in checkpoints:
        logging.info("Evaluating seed=%s checkpoint=%s", seed, checkpoint_path)
        checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id)
        class_to_task, _ = build_task_metadata(data_manager, model._cur_task, model._total_classes)
        loader = build_test_loader(
            data_manager=data_manager,
            total_classes=model._total_classes,
            batch_size=args["batch_size"],
            num_workers=cli_args.num_workers,
        )
        metrics = eval_route_abc(
            model=model,
            loader=loader,
            class_to_task=class_to_task,
            apply_scale=not cli_args.no_scale,
        )
        seed_results.append(
            {
                "task_id": model._cur_task,
                "known_classes": model._known_classes,
                "total_classes": model._total_classes,
                "num_experts": model._cur_task + 1,
                "checkpoint": str(checkpoint_path),
                "checkpoint_full_cnn": checkpoint.get("cnn_accy"),
                "applied_scale": not cli_args.no_scale,
                "routing_rule": "argmax_expert(max_class_logit_of_expert), matching models/tunamax.py::_eval_cnn",
                **metrics,
            }
        )
        logging.info(
            "Task %s: A=%.4f B=%.4f C=%.4f full=%.4f",
            model._cur_task,
            metrics["A_task_id_accuracy"],
            metrics["B_acc_given_task_id_correct"],
            metrics["C_acc_given_task_id_wrong"],
            metrics["full_acc_direct"],
        )

    return {
        "seed": seed,
        "model_name": args["model_name"],
        "dataset": args["dataset"],
        "results": seed_results,
    }


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [spie_v2_route_abc_eval] %(message)s")
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
