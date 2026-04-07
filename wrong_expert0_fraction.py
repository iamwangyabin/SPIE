import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from expert_response_analysis import (
    build_task_metadata,
    build_test_loader,
    get_logits_for_expert,
    is_incremental_expert_model,
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
            "Compute the fraction of wrong test samples routed to a target expert, "
            "plus the target-vs-true-expert max-score margin on stolen wrong samples."
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
        help="Optional direct checkpoint path. If set, --checkpoint-dir is not used for locating the checkpoint.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="latest",
        help="Task checkpoint to analyze: latest, final, last, or an integer task id. Default: latest.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Analyze only this seed. Defaults to all config seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for analysis.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for the test loader.")
    parser.add_argument("--expert-id", type=int, default=0, help="Expert id to count. Default: 0.")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply Tuna/TunaMax/OnlyMax-style cosine scores by args['scale'].",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save the summary JSON.")
    return parser


def active_expert_ids(model) -> Sequence[int]:
    return list(range(model._cur_task + 1))


def map_local_routes(local_routes: torch.Tensor, active_ids: Sequence[int]) -> torch.Tensor:
    route_lookup = torch.tensor(active_ids, device=local_routes.device, dtype=torch.long)
    return route_lookup[local_routes]


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "p99": float(torch.quantile(tensor, 0.99).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


@torch.no_grad()
def max_scores_by_expert(
    model,
    inputs: torch.Tensor,
    expert_ids: Sequence[int],
    task_offsets: Sequence[int],
    apply_scale: bool,
) -> torch.Tensor:
    scores = []
    for expert_id in expert_ids:
        logits, _ = get_logits_for_expert(
            model=model,
            inputs=inputs,
            expert_id=expert_id,
            total_classes=model._total_classes,
            task_offsets=task_offsets,
            apply_scale=apply_scale,
        )
        scores.append(logits.max(dim=1).values)
    return torch.stack(scores, dim=1)


@torch.no_grad()
def predict_spie(model, inputs: torch.Tensor, active_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = model._network.predict_all(inputs, active_experts=active_ids)
    top1_concat = pred["logits"].argmax(dim=1)
    mapping = pred["mapping"]
    pred_classes = torch.tensor(
        [
            model.task_offsets[mapping[int(idx)]["expert_idx"]] + mapping[int(idx)]["local_class_idx"]
            for idx in top1_concat.detach().cpu().tolist()
        ],
        device=inputs.device,
        dtype=torch.long,
    )
    routed_experts = pred["pred_expert_idx"].to(device=inputs.device, dtype=torch.long)
    return pred_classes, routed_experts


@torch.no_grad()
def predict_onlymax(
    model,
    inputs: torch.Tensor,
    active_ids: Sequence[int],
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model._network.predict_onlymax(
        inputs,
        active_experts=active_ids,
        total_classes=model._total_classes,
    )
    logits = out["selected_logits"]
    if apply_scale:
        logits = logits * float(model.args.get("scale", 1.0))
    pred_classes = logits.argmax(dim=1)
    routed_experts = map_local_routes(out["best_expert_per_sample"].long(), active_ids)
    return pred_classes, routed_experts


@torch.no_grad()
def predict_tunamax_style(
    model,
    inputs: torch.Tensor,
    active_ids: Sequence[int],
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_per_expert = []
    for expert_id in active_ids:
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
        logits = model._network.fc(features)["logits"][:, : model._total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        logits_per_expert.append(logits)

    stacked_logits = torch.stack(logits_per_expert, dim=0)
    best_scores_per_expert = stacked_logits.max(dim=2).values.transpose(0, 1)
    best_local_expert = best_scores_per_expert.argmax(dim=1)
    routed_experts = map_local_routes(best_local_expert, active_ids)
    selected_logits = stacked_logits.permute(1, 0, 2)[
        torch.arange(stacked_logits.shape[1], device=stacked_logits.device),
        best_local_expert,
    ]
    pred_classes = selected_logits.argmax(dim=1)
    return pred_classes, routed_experts


@torch.no_grad()
def predict_tuna_entropy(
    model,
    inputs: torch.Tensor,
    active_ids: Sequence[int],
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model._cur_task == 0:
        features = model._network.backbone(inputs, adapter_id=0, train=False)["features"]
        logits = model._network.fc(features)["logits"][:, : model._total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        return logits.argmax(dim=1), torch.zeros(inputs.shape[0], device=inputs.device, dtype=torch.long)

    all_entropies = []
    all_logits = []
    for expert_id in active_ids:
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
        logits = model._network.fc(features)["logits"][:, : model._total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        all_entropies.append(entropy)
        all_logits.append(logits)

    entropies = torch.stack(all_entropies, dim=0)
    stacked_logits = torch.stack(all_logits, dim=0)
    best_local_expert = torch.argmin(entropies, dim=0)
    routed_experts = map_local_routes(best_local_expert, active_ids)
    selected_logits = stacked_logits.permute(1, 0, 2)[
        torch.arange(stacked_logits.shape[1], device=stacked_logits.device),
        best_local_expert,
    ]

    features = model._network.backbone(inputs, adapter_id=model._cur_task + 1, train=False)["features"]
    general_logits = model._network.fc(features)["logits"][:, : model._total_classes]
    if apply_scale:
        general_logits = general_logits * float(model.args.get("scale", 1.0))

    outputs = F.softmax(selected_logits, dim=1) + F.softmax(general_logits, dim=1)
    pred_classes = outputs.argmax(dim=1)
    return pred_classes, routed_experts


@torch.no_grad()
def predict_with_routes(
    model,
    inputs: torch.Tensor,
    active_ids: Sequence[int],
    model_name: str,
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    network = model._network
    model_name = model_name.lower()

    if is_incremental_expert_model(network):
        pred_classes, routed_experts = predict_spie(model, inputs, active_ids)
        return pred_classes, routed_experts, "top1 class owner in concatenated expert logits"

    if hasattr(network, "predict_onlymax"):
        pred_classes, routed_experts = predict_onlymax(model, inputs, active_ids, apply_scale)
        return pred_classes, routed_experts, "expert with the largest per-sample max logit"

    if model_name == "tuna":
        pred_classes, routed_experts = predict_tuna_entropy(model, inputs, active_ids, apply_scale)
        return pred_classes, routed_experts, "adapter with minimum entropy before TUNA general-branch fusion"

    if hasattr(network, "backbone") and hasattr(network, "fc"):
        pred_classes, routed_experts = predict_tunamax_style(model, inputs, active_ids, apply_scale)
        return pred_classes, routed_experts, "expert/adapter with the largest per-sample max logit"

    raise TypeError(f"Unsupported network type: {type(network).__name__}")


def collect_wrong_route_fraction(
    model,
    loader,
    expert_id: int,
    class_to_task: Dict[int, int],
    task_offsets: Sequence[int],
    model_name: str,
    apply_scale: bool,
) -> Dict:
    model._network.eval()
    expert_ids = active_expert_ids(model)
    if expert_id not in expert_ids:
        raise ValueError(f"expert_id={expert_id} is not active for task {model._cur_task}; active experts={expert_ids}.")

    route_counts = {int(idx): 0 for idx in expert_ids}
    wrong_route_counts = {int(idx): 0 for idx in expert_ids}
    total = 0
    wrong_total = 0
    wrong_to_expert = 0
    stolen_wrong_to_expert = 0
    target_beats_true_count = 0
    target_true_margins = []
    route_definition: Optional[str] = None

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        pred_classes, routed_experts, route_definition = predict_with_routes(
            model=model,
            inputs=inputs,
            active_ids=expert_ids,
            model_name=model_name,
            apply_scale=apply_scale,
        )

        wrong_mask = pred_classes.ne(targets)
        expert_mask = routed_experts.eq(expert_id)
        true_experts = torch.tensor(
            [class_to_task[int(target)] for target in targets.detach().cpu().tolist()],
            device=targets.device,
            dtype=torch.long,
        )
        stolen_wrong_mask = wrong_mask & expert_mask & true_experts.ne(expert_id)

        total += int(targets.numel())
        wrong_total += int(wrong_mask.sum().item())
        wrong_to_expert += int((wrong_mask & expert_mask).sum().item())
        stolen_wrong_count = int(stolen_wrong_mask.sum().item())
        stolen_wrong_to_expert += stolen_wrong_count

        if stolen_wrong_count > 0:
            score_matrix = max_scores_by_expert(
                model=model,
                inputs=inputs,
                expert_ids=expert_ids,
                task_offsets=task_offsets,
                apply_scale=apply_scale,
            )
            active_lookup = {int(active_id): idx for idx, active_id in enumerate(expert_ids)}
            target_local_expert = active_lookup[int(expert_id)]
            true_local_experts = torch.tensor(
                [active_lookup[int(true_expert)] for true_expert in true_experts.detach().cpu().tolist()],
                device=targets.device,
                dtype=torch.long,
            )
            batch_indices = torch.arange(targets.shape[0], device=targets.device)
            target_scores = score_matrix[:, target_local_expert]
            true_scores = score_matrix[batch_indices, true_local_experts]
            margins = (target_scores - true_scores)[stolen_wrong_mask]
            target_beats_true_count += int((margins > 0).sum().item())
            target_true_margins.extend(float(value) for value in margins.detach().cpu().tolist())

        for route_id in torch.unique(routed_experts).detach().cpu().tolist():
            route_mask = routed_experts.eq(int(route_id))
            route_counts[int(route_id)] = route_counts.get(int(route_id), 0) + int(route_mask.sum().item())
            wrong_route_counts[int(route_id)] = wrong_route_counts.get(int(route_id), 0) + int(
                (wrong_mask & route_mask).sum().item()
            )

    target_suffix = f"expert_{expert_id}"
    margin_summary = summarize_values(target_true_margins)
    return {
        "total_samples": total,
        "wrong_samples": wrong_total,
        "target_expert": expert_id,
        "wrong_samples_routed_to_target_expert": wrong_to_expert,
        "fraction_of_wrong_samples_routed_to_target_expert": (
            wrong_to_expert / wrong_total if wrong_total > 0 else float("nan")
        ),
        f"wrong_samples_routed_to_{target_suffix}": wrong_to_expert,
        f"fraction_of_wrong_samples_routed_to_{target_suffix}": (
            wrong_to_expert / wrong_total if wrong_total > 0 else float("nan")
        ),
        "fraction_of_all_samples_wrong_and_routed_to_target_expert": (
            wrong_to_expert / total if total > 0 else float("nan")
        ),
        "top1_accuracy": 1.0 - wrong_total / total if total > 0 else float("nan"),
        "top1_accuracy_percent": (1.0 - wrong_total / total) * 100.0 if total > 0 else float("nan"),
        "route_counts": route_counts,
        "wrong_route_counts": wrong_route_counts,
        "stolen_wrong_samples_routed_to_target_expert": stolen_wrong_to_expert,
        f"stolen_wrong_samples_routed_to_{target_suffix}": stolen_wrong_to_expert,
        "target_minus_true_expert_max_score": margin_summary,
        f"{target_suffix}_minus_true_expert_max_score": margin_summary,
        "target_expert_beats_true_expert_count": target_beats_true_count,
        f"{target_suffix}_beats_true_expert_count": target_beats_true_count,
        "fraction_of_stolen_wrong_target_expert_samples_where_target_beats_true_expert": (
            target_beats_true_count / stolen_wrong_to_expert if stolen_wrong_to_expert > 0 else float("nan")
        ),
        f"fraction_of_stolen_wrong_{target_suffix}_samples_where_{target_suffix}_beats_true_expert": (
            target_beats_true_count / stolen_wrong_to_expert if stolen_wrong_to_expert > 0 else float("nan")
        ),
        "margin_definition": (
            "For wrong samples routed to target_expert whose true expert differs from target_expert, "
            "max_c z_target_expert,c(x) - max_c z_true_expert,c(x)."
        ),
        "route_definition": route_definition,
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

    checkpoint_dir = None if cli_args.checkpoint else resolve_checkpoint_dir(args)
    if cli_args.checkpoint:
        checkpoint_path = Path(cli_args.checkpoint)
        expected_task_id = None
        if str(cli_args.task_id).strip().lower() != "latest":
            expected_task_id = resolve_task_id(cli_args.task_id, checkpoint_dir, data_manager)
    else:
        expected_task_id = resolve_task_id(cli_args.task_id, checkpoint_dir, data_manager)
        checkpoint_path = checkpoint_dir / f"task_{expected_task_id}.pkl"

    model = factory.get_model(args["model_name"], args)
    checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id)
    class_to_task, task_offsets = build_task_metadata(data_manager, model._cur_task, model._total_classes)
    loader = build_test_loader(
        data_manager=data_manager,
        total_classes=model._total_classes,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
    )

    logging.info(
        "Analyzing seed=%s task=%s checkpoint=%s total_classes=%s",
        seed,
        model._cur_task,
        checkpoint_path,
        model._total_classes,
    )
    summary = collect_wrong_route_fraction(
        model=model,
        loader=loader,
        expert_id=cli_args.expert_id,
        class_to_task=class_to_task,
        task_offsets=task_offsets,
        model_name=args["model_name"],
        apply_scale=not cli_args.no_scale,
    )
    summary.update(
        {
            "seed": seed,
            "model_name": args["model_name"],
            "dataset": args["dataset"],
            "task_id": model._cur_task,
            "known_classes": model._known_classes,
            "total_classes": model._total_classes,
            "checkpoint": str(checkpoint_path),
            "checkpoint_cnn_accy": checkpoint.get("cnn_accy"),
            "applied_scale": not cli_args.no_scale,
        }
    )
    return summary


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [wrong_expert0_fraction] %(message)s")
    config = load_json(cli_args.config)

    seeds = iter_config_seeds(config, cli_args.seed)
    if cli_args.checkpoint and len(seeds) > 1 and cli_args.seed is None:
        logging.warning("--checkpoint was set with multiple config seeds; analyzing the first seed only.")
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
