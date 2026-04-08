import argparse
import copy
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from expert_response_analysis import (
    build_task_metadata,
    build_test_loader,
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


SUPPORTED_MODEL_NAMES = {"spie_v5", "spiev5"}


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test the final SPiE v5 routing accuracy and whether samples are biased "
            "toward particular experts."
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
        help="Optional direct checkpoint path. If set, --checkpoint-dir is not used.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="latest",
        help="Checkpoint to test: latest, final, last, or an integer task id. Default: latest.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Test only this seed. Defaults to all config seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for this test.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for the test loader.")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply logits by args['scale'] before routing/class prediction.",
    )
    parser.add_argument(
        "--bias-threshold",
        type=float,
        default=0.05,
        help=(
            "Heuristic threshold for flagging distribution bias. The flag is true when "
            "0.5 * sum(abs(routed_fraction - true_task_fraction)) is larger than this value."
        ),
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save the summary JSON.")
    return parser


def validate_config(config: Dict) -> None:
    model_name = str(config.get("model_name", "")).lower()
    if model_name not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "spie_v5_route_bias_test.py expects a SPiE v5 config with "
            f"model_name in {sorted(SUPPORTED_MODEL_NAMES)}, got {config.get('model_name')!r}."
        )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


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


def expected_task_for_direct_checkpoint(task_id_arg: str, data_manager: DataManager) -> Optional[int]:
    value = str(task_id_arg).strip().lower()
    if value == "latest":
        return None
    if value in {"final", "last"}:
        return data_manager.nb_tasks - 1
    return int(value)


def resolve_checkpoint(args: Dict, cli_args: argparse.Namespace, data_manager: DataManager) -> Tuple[Path, Optional[int]]:
    if cli_args.checkpoint:
        checkpoint_path = Path(cli_args.checkpoint)
        expected_task_id = expected_task_for_direct_checkpoint(cli_args.task_id, data_manager)
        return checkpoint_path, expected_task_id

    checkpoint_dir = resolve_checkpoint_dir(args)
    task_id = resolve_task_id(cli_args.task_id, checkpoint_dir, data_manager)
    return checkpoint_dir / f"task_{task_id}.pkl", task_id


def map_true_tasks(targets: torch.Tensor, class_to_task: Dict[int, int]) -> torch.Tensor:
    return torch.tensor(
        [class_to_task[int(target)] for target in targets.detach().cpu().tolist()],
        device=targets.device,
        dtype=torch.long,
    )


@torch.no_grad()
def route_spie_v5_batch(
    model,
    inputs: torch.Tensor,
    expert_ids: Sequence[int],
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_per_expert = []
    for expert_id in expert_ids:
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
        logits = model._network.fc(features)["logits"][:, : model._total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        logits_per_expert.append(logits)

    stacked_logits = torch.stack(logits_per_expert, dim=0)
    route_scores = stacked_logits.max(dim=2).values.transpose(0, 1)
    best_local_expert = route_scores.argmax(dim=1)
    selected_logits = stacked_logits.permute(1, 0, 2)[
        torch.arange(stacked_logits.shape[1], device=stacked_logits.device),
        best_local_expert,
    ]
    pred_classes = selected_logits.argmax(dim=1)
    expert_lookup = torch.tensor(expert_ids, device=inputs.device, dtype=torch.long)
    routed_experts = expert_lookup[best_local_expert]
    return pred_classes, routed_experts, route_scores


def tensor_matrix_to_rows(matrix: torch.Tensor) -> List[List[int]]:
    return [[int(value) for value in row] for row in matrix.tolist()]


def tensor_matrix_to_fraction_rows(matrix: torch.Tensor) -> List[List[float]]:
    rows = []
    for row in matrix.tolist():
        row_total = float(sum(row))
        rows.append([safe_divide(value, row_total) for value in row])
    return rows


def score_stats(score_sum: float, score_sq_sum: float, count: int) -> Dict[str, float]:
    if count == 0:
        return {"mean": float("nan"), "std": float("nan")}
    mean = score_sum / float(count)
    variance = max(score_sq_sum / float(count) - mean * mean, 0.0)
    return {"mean": float(mean), "std": float(math.sqrt(variance))}


def distribution_entropy(fractions: Sequence[float]) -> float:
    entropy = 0.0
    for fraction in fractions:
        if fraction > 0:
            entropy -= fraction * math.log(fraction)
    return float(entropy)


def build_per_expert_summary(
    expert_ids: Sequence[int],
    confusion: torch.Tensor,
    routed_counts: torch.Tensor,
    true_task_counts: torch.Tensor,
    route_score_sums: torch.Tensor,
    route_score_sq_sums: torch.Tensor,
    total_samples: int,
) -> List[Dict]:
    rows = []
    for local_idx, expert_id in enumerate(expert_ids):
        routed_count = int(routed_counts[local_idx].item())
        true_count = int(true_task_counts[local_idx].item())
        diagonal_count = int(confusion[local_idx, local_idx].item())
        routed_fraction = safe_divide(routed_count, total_samples)
        true_fraction = safe_divide(true_count, total_samples)
        stats = score_stats(
            score_sum=float(route_score_sums[local_idx].item()),
            score_sq_sum=float(route_score_sq_sums[local_idx].item()),
            count=total_samples,
        )
        rows.append(
            {
                "expert_id": int(expert_id),
                "true_task_count": true_count,
                "true_task_fraction": true_fraction,
                "routed_count": routed_count,
                "routed_fraction": routed_fraction,
                "routed_fraction_minus_true_task_fraction": routed_fraction - true_fraction,
                "route_recall_for_true_task": safe_divide(diagonal_count, true_count),
                "route_precision_when_selected": safe_divide(diagonal_count, routed_count),
                "mean_route_score": stats["mean"],
                "std_route_score": stats["std"],
            }
        )
    return rows


def build_true_task_summary(
    expert_ids: Sequence[int],
    confusion: torch.Tensor,
    true_task_counts: torch.Tensor,
) -> List[Dict]:
    rows = []
    for local_idx, expert_id in enumerate(expert_ids):
        true_count = int(true_task_counts[local_idx].item())
        correct_count = int(confusion[local_idx, local_idx].item())
        if true_count > 0:
            top_local_route = int(confusion[local_idx].argmax().item())
            top_count = int(confusion[local_idx, top_local_route].item())
            top_expert = int(expert_ids[top_local_route])
            top_fraction = safe_divide(top_count, true_count)
        else:
            top_expert = -1
            top_count = 0
            top_fraction = float("nan")
        rows.append(
            {
                "true_task": int(expert_id),
                "count": true_count,
                "route_correct_count": correct_count,
                "route_accuracy": safe_divide(correct_count, true_count),
                "most_common_routed_expert": top_expert,
                "most_common_routed_count": top_count,
                "most_common_routed_fraction": top_fraction,
            }
        )
    return rows


def build_bias_summary(
    per_expert: Sequence[Dict],
    total_variation_distance: float,
    bias_threshold: float,
) -> Dict:
    if not per_expert:
        return {}

    most_selected = max(per_expert, key=lambda row: row["routed_fraction"])
    least_selected = min(per_expert, key=lambda row: row["routed_fraction"])
    most_over_selected = max(per_expert, key=lambda row: row["routed_fraction_minus_true_task_fraction"])
    most_under_selected = min(per_expert, key=lambda row: row["routed_fraction_minus_true_task_fraction"])
    fractions = [float(row["routed_fraction"]) for row in per_expert]
    entropy = distribution_entropy(fractions)
    normalized_entropy = safe_divide(entropy, math.log(len(fractions))) if len(fractions) > 1 else 1.0
    min_count = int(least_selected["routed_count"])
    max_count = int(most_selected["routed_count"])

    return {
        "bias_reference": "true_task_fraction",
        "bias_threshold": float(bias_threshold),
        "total_variation_distance_vs_true_task_distribution": float(total_variation_distance),
        "has_distribution_bias_vs_true_task_distribution": bool(total_variation_distance > bias_threshold),
        "most_selected_expert": int(most_selected["expert_id"]),
        "most_selected_routed_fraction": float(most_selected["routed_fraction"]),
        "least_selected_expert": int(least_selected["expert_id"]),
        "least_selected_routed_fraction": float(least_selected["routed_fraction"]),
        "most_over_selected_expert": int(most_over_selected["expert_id"]),
        "most_over_selected_fraction_excess": float(
            most_over_selected["routed_fraction_minus_true_task_fraction"]
        ),
        "most_under_selected_expert": int(most_under_selected["expert_id"]),
        "most_under_selected_fraction_deficit": float(
            most_under_selected["routed_fraction_minus_true_task_fraction"]
        ),
        "max_to_min_routed_count_ratio": float("inf") if min_count == 0 else max_count / float(min_count),
        "routed_distribution_entropy": entropy,
        "routed_distribution_normalized_entropy": normalized_entropy,
    }


@torch.no_grad()
def collect_route_bias(model, loader, class_to_task: Dict[int, int], apply_scale: bool, bias_threshold: float) -> Dict:
    model._network.eval()
    expert_ids = list(range(model._cur_task + 1))
    num_experts = len(expert_ids)
    active_lookup = {int(expert_id): local_idx for local_idx, expert_id in enumerate(expert_ids)}

    confusion = torch.zeros((num_experts, num_experts), dtype=torch.long)
    routed_counts = torch.zeros(num_experts, dtype=torch.long)
    true_task_counts = torch.zeros(num_experts, dtype=torch.long)
    route_score_sums = torch.zeros(num_experts, dtype=torch.float64)
    route_score_sq_sums = torch.zeros(num_experts, dtype=torch.float64)
    total_samples = 0
    route_correct = 0
    class_correct = 0

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        pred_classes, routed_experts, route_scores = route_spie_v5_batch(
            model=model,
            inputs=inputs,
            expert_ids=expert_ids,
            apply_scale=apply_scale,
        )
        true_experts = map_true_tasks(targets, class_to_task)
        true_local = torch.tensor(
            [active_lookup[int(expert_id)] for expert_id in true_experts.detach().cpu().tolist()],
            device=targets.device,
            dtype=torch.long,
        )
        routed_local = torch.tensor(
            [active_lookup[int(expert_id)] for expert_id in routed_experts.detach().cpu().tolist()],
            device=targets.device,
            dtype=torch.long,
        )

        flat_confusion = true_local * num_experts + routed_local
        confusion += torch.bincount(flat_confusion, minlength=num_experts * num_experts).reshape(
            num_experts, num_experts
        ).cpu()
        routed_counts += torch.bincount(routed_local, minlength=num_experts).cpu()
        true_task_counts += torch.bincount(true_local, minlength=num_experts).cpu()
        route_score_sums += route_scores.detach().double().sum(dim=0).cpu()
        route_score_sq_sums += route_scores.detach().double().pow(2).sum(dim=0).cpu()

        route_correct += int(routed_experts.eq(true_experts).sum().item())
        class_correct += int(pred_classes.eq(targets).sum().item())
        total_samples += int(targets.numel())

    per_expert = build_per_expert_summary(
        expert_ids=expert_ids,
        confusion=confusion,
        routed_counts=routed_counts,
        true_task_counts=true_task_counts,
        route_score_sums=route_score_sums,
        route_score_sq_sums=route_score_sq_sums,
        total_samples=total_samples,
    )
    route_fractions = [row["routed_fraction"] for row in per_expert]
    true_fractions = [row["true_task_fraction"] for row in per_expert]
    total_variation_distance = 0.5 * sum(abs(a - b) for a, b in zip(route_fractions, true_fractions))

    return {
        "num_samples": total_samples,
        "num_experts": num_experts,
        "route_correct": route_correct,
        "route_accuracy": safe_divide(route_correct, total_samples),
        "route_accuracy_percent": safe_divide(route_correct, total_samples) * 100.0,
        "class_correct_after_routing": class_correct,
        "class_accuracy_after_routing": safe_divide(class_correct, total_samples),
        "class_accuracy_after_routing_percent": safe_divide(class_correct, total_samples) * 100.0,
        "per_expert": per_expert,
        "per_true_task": build_true_task_summary(
            expert_ids=expert_ids,
            confusion=confusion,
            true_task_counts=true_task_counts,
        ),
        "confusion_matrix": {
            "rows_true_task": [int(expert_id) for expert_id in expert_ids],
            "cols_routed_expert": [int(expert_id) for expert_id in expert_ids],
            "counts": tensor_matrix_to_rows(confusion),
            "row_fractions": tensor_matrix_to_fraction_rows(confusion),
        },
        "bias": build_bias_summary(
            per_expert=per_expert,
            total_variation_distance=total_variation_distance,
            bias_threshold=bias_threshold,
        ),
        "routing_rule": "argmax_expert(max_class_logit_of_expert), matching models/tunamax.py::_eval_cnn",
    }


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
    checkpoint_path, expected_task_id = resolve_checkpoint(args, cli_args, data_manager)

    logging.info("Testing seed=%s checkpoint=%s", seed, checkpoint_path)
    checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id)
    class_to_task, _ = build_task_metadata(data_manager, model._cur_task, model._total_classes)
    loader = build_test_loader(
        data_manager=data_manager,
        total_classes=model._total_classes,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
    )
    metrics = collect_route_bias(
        model=model,
        loader=loader,
        class_to_task=class_to_task,
        apply_scale=not cli_args.no_scale,
        bias_threshold=cli_args.bias_threshold,
    )
    return {
        "seed": seed,
        "model_name": args["model_name"],
        "dataset": args["dataset"],
        "task_id": model._cur_task,
        "known_classes": model._known_classes,
        "total_classes": model._total_classes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_full_cnn": checkpoint.get("cnn_accy"),
        "applied_scale": not cli_args.no_scale,
        **metrics,
    }


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [spie_v5_route_bias_test] %(message)s")
    config = load_json(cli_args.config)
    validate_config(config)

    seeds = iter_config_seeds(config, cli_args.seed)
    if cli_args.checkpoint and len(seeds) > 1 and cli_args.seed is None:
        logging.warning("--checkpoint was set with multiple config seeds; testing the first seed only.")
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
