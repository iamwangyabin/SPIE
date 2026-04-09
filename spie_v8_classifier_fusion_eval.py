import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from expert_response_analysis import (
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
from utils.toolkit import accuracy


DEFAULT_STRATEGIES = [
    "baseline_max_logit_route",
    "route_max_prob",
    "route_min_entropy",
    "mean_logits",
    "mean_probs",
    "weighted_logits_max_logit",
    "weighted_probs_max_logit",
    "weighted_probs_neg_entropy",
    "topk_mean_logits_max_logit",
    "topk_mean_probs_max_logit",
    "majority_vote_top1",
    "baseline_max_logit_task_masked",
]


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multiple classifier-fusion rules on a trained SPiE v8 / TunaMax-style checkpoint. "
            "The baseline matches the current per-sample max-logit expert routing."
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
        help="Checkpoint to evaluate: latest, final, last, or an integer task id. Default: latest.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Evaluate only this seed. Defaults to all config seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for the test loader.")
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help="Comma-separated strategy names, or 'all'.",
    )
    parser.add_argument(
        "--topk-experts",
        type=int,
        default=2,
        help="Number of experts to average for topk_mean_* strategies. Default: 2.",
    )
    parser.add_argument(
        "--score-temperature",
        type=float,
        default=1.0,
        help="Temperature for max-logit-based soft expert weighting. Default: 1.0.",
    )
    parser.add_argument(
        "--entropy-temperature",
        type=float,
        default=1.0,
        help="Temperature for entropy-based soft expert weighting. Default: 1.0.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply cosine scores by args['scale'].",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save the summary JSON.")
    return parser


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


def resolve_strategy_names(raw: str) -> List[str]:
    value = str(raw).strip().lower()
    if value == "all":
        return list(DEFAULT_STRATEGIES)

    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Available: {DEFAULT_STRATEGIES}")
    if "baseline_max_logit_route" not in requested:
        requested = ["baseline_max_logit_route"] + requested
    return requested


def active_expert_ids(model) -> List[int]:
    return list(range(model._cur_task + 1))


def build_expert_class_masks(data_manager: DataManager, task_id: int, total_classes: int) -> torch.Tensor:
    masks = []
    class_offset = 0
    for expert_id in range(task_id + 1):
        task_size = data_manager.get_task_size(expert_id)
        mask = torch.zeros(total_classes, dtype=torch.bool)
        upper = min(class_offset + task_size, total_classes)
        mask[class_offset:upper] = True
        masks.append(mask)
        class_offset += task_size
    return torch.stack(masks, dim=0)


@torch.no_grad()
def stack_expert_logits(
    model,
    inputs: torch.Tensor,
    expert_ids: Sequence[int],
    apply_scale: bool,
) -> torch.Tensor:
    logits_per_expert = []
    for expert_id in expert_ids:
        features = model._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
        logits = model._network.fc(features)["logits"][:, : model._total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        logits_per_expert.append(logits)
    return torch.stack(logits_per_expert, dim=0)


def select_scores_by_expert(stacked_scores: torch.Tensor, local_expert_idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_classes = stacked_scores.shape[1], stacked_scores.shape[2]
    scores_bec = stacked_scores.permute(1, 0, 2)
    gather_idx = local_expert_idx.view(batch_size, 1, 1).expand(-1, 1, num_classes)
    return scores_bec.gather(1, gather_idx).squeeze(1)


def topk_average_scores(
    stacked_scores: torch.Tensor,
    expert_rank_scores: torch.Tensor,
    topk_experts: int,
) -> torch.Tensor:
    num_experts = stacked_scores.shape[0]
    topk_experts = max(1, min(int(topk_experts), num_experts))
    top_idx = torch.topk(expert_rank_scores, k=topk_experts, dim=0, largest=True, sorted=True).indices
    scores_bec = stacked_scores.permute(1, 0, 2)
    gather_idx = top_idx.transpose(0, 1).unsqueeze(-1).expand(-1, -1, stacked_scores.shape[2])
    selected_scores = scores_bec.gather(1, gather_idx)
    return selected_scores.mean(dim=1)


def weighted_average_scores(stacked_scores: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    return (expert_weights.unsqueeze(-1) * stacked_scores).sum(dim=0)


def one_hot_vote_scores(top1_classes: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    num_experts, batch_size, num_classes = probs.shape
    vote_scores = probs.sum(dim=0) * 1e-3
    for expert_idx in range(num_experts):
        vote_scores.scatter_add_(
            dim=1,
            index=top1_classes[expert_idx].unsqueeze(1),
            src=torch.ones((batch_size, 1), device=vote_scores.device, dtype=vote_scores.dtype),
        )
    return vote_scores


def apply_task_mask(selected_scores: torch.Tensor, local_expert_idx: torch.Tensor, expert_class_masks: torch.Tensor) -> torch.Tensor:
    mask = expert_class_masks.to(selected_scores.device)[local_expert_idx]
    return selected_scores.masked_fill(~mask, torch.finfo(selected_scores.dtype).min)


def strategy_scores(
    stacked_logits: torch.Tensor,
    expert_class_masks: torch.Tensor,
    topk_experts: int,
    score_temperature: float,
    entropy_temperature: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    probs = F.softmax(stacked_logits, dim=2)
    max_logits = stacked_logits.max(dim=2).values
    max_probs = probs.max(dim=2).values
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=2)
    top1_classes = stacked_logits.argmax(dim=2)

    best_by_max_logit = max_logits.argmax(dim=0)
    best_by_max_prob = max_probs.argmax(dim=0)
    best_by_min_entropy = entropy.argmin(dim=0)

    score_temp = max(float(score_temperature), 1e-8)
    entropy_temp = max(float(entropy_temperature), 1e-8)
    max_logit_weights = F.softmax(max_logits / score_temp, dim=0)
    neg_entropy_weights = F.softmax((-entropy) / entropy_temp, dim=0)

    selected_logits_max_logit = select_scores_by_expert(stacked_logits, best_by_max_logit)
    selected_logits_max_prob = select_scores_by_expert(stacked_logits, best_by_max_prob)
    selected_logits_min_entropy = select_scores_by_expert(stacked_logits, best_by_min_entropy)

    strategy_to_scores = {
        "baseline_max_logit_route": selected_logits_max_logit,
        "route_max_prob": selected_logits_max_prob,
        "route_min_entropy": selected_logits_min_entropy,
        "mean_logits": stacked_logits.mean(dim=0),
        "mean_probs": probs.mean(dim=0),
        "weighted_logits_max_logit": weighted_average_scores(stacked_logits, max_logit_weights),
        "weighted_probs_max_logit": weighted_average_scores(probs, max_logit_weights),
        "weighted_probs_neg_entropy": weighted_average_scores(probs, neg_entropy_weights),
        "topk_mean_logits_max_logit": topk_average_scores(stacked_logits, max_logits, topk_experts),
        "topk_mean_probs_max_logit": topk_average_scores(probs, max_logits, topk_experts),
        "majority_vote_top1": one_hot_vote_scores(top1_classes, probs),
        "baseline_max_logit_task_masked": apply_task_mask(
            selected_logits_max_logit,
            best_by_max_logit,
            expert_class_masks,
        ),
    }

    route_info = {
        "baseline_max_logit_route": best_by_max_logit,
        "route_max_prob": best_by_max_prob,
        "route_min_entropy": best_by_min_entropy,
        "baseline_max_logit_task_masked": best_by_max_logit,
    }
    return strategy_to_scores, route_info


def finalize_predictions(predicts: torch.Tensor, requested_topk: int) -> torch.Tensor:
    topk = predicts.shape[1]
    if topk >= requested_topk:
        return predicts[:, :requested_topk]
    pad = torch.full(
        (predicts.shape[0], requested_topk - topk),
        -1,
        device=predicts.device,
        dtype=predicts.dtype,
    )
    return torch.cat([predicts, pad], dim=1)


def route_histogram(local_routes: torch.Tensor, active_ids: Sequence[int]) -> Dict[str, int]:
    counts = {}
    for local_idx, expert_id in enumerate(active_ids):
        counts[str(expert_id)] = int(local_routes.eq(local_idx).sum().item())
    return counts


def evaluate_fusion_strategies(
    model,
    loader,
    strategy_names: Sequence[str],
    expert_class_masks: torch.Tensor,
    topk_experts: int,
    score_temperature: float,
    entropy_temperature: float,
    apply_scale: bool,
) -> Dict[str, Dict]:
    model._network.eval()
    active_ids = active_expert_ids(model)
    y_true = []
    prediction_bank = {name: [] for name in strategy_names}
    route_bank = {name: [] for name in strategy_names if name in {
        "baseline_max_logit_route",
        "route_max_prob",
        "route_min_entropy",
        "baseline_max_logit_task_masked",
    }}

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        stacked_logits = stack_expert_logits(
            model=model,
            inputs=inputs,
            expert_ids=active_ids,
            apply_scale=apply_scale,
        )
        batch_scores, batch_routes = strategy_scores(
            stacked_logits=stacked_logits,
            expert_class_masks=expert_class_masks,
            topk_experts=topk_experts,
            score_temperature=score_temperature,
            entropy_temperature=entropy_temperature,
        )

        y_true.append(targets.numpy())
        for strategy_name in strategy_names:
            score_matrix = batch_scores[strategy_name]
            predicts = torch.topk(
                score_matrix,
                k=min(model.topk, score_matrix.shape[1]),
                dim=1,
                largest=True,
                sorted=True,
            )[1]
            predicts = finalize_predictions(predicts, model.topk)
            prediction_bank[strategy_name].append(predicts.cpu().numpy())
            if strategy_name in route_bank:
                route_bank[strategy_name].append(batch_routes[strategy_name].cpu())

    y_true = np.concatenate(y_true)
    results = {}
    for strategy_name in strategy_names:
        y_pred = np.concatenate(prediction_bank[strategy_name])
        metrics = evaluate_predictions(model, y_pred, y_true)
        result = {
            "metrics": metrics,
        }
        if strategy_name in route_bank:
            local_routes = torch.cat(route_bank[strategy_name], dim=0)
            result["route_histogram"] = route_histogram(local_routes, active_ids)
        results[strategy_name] = result
    return results


def checkpoint_spec(
    args: Dict,
    cli_args: argparse.Namespace,
    data_manager: DataManager,
) -> Tuple[Path, Optional[int]]:
    if cli_args.checkpoint:
        checkpoint_path = Path(cli_args.checkpoint)
        task_arg = str(cli_args.task_id).strip().lower()
        if task_arg in {"latest", "final", "last"}:
            expected_task_id = None if task_arg == "latest" else data_manager.nb_tasks - 1
        else:
            expected_task_id = int(task_arg)
        return checkpoint_path, expected_task_id

    checkpoint_dir = resolve_checkpoint_dir(args)
    expected_task_id = resolve_task_id(cli_args.task_id, checkpoint_dir, data_manager)
    return checkpoint_dir / f"task_{expected_task_id}.pkl", expected_task_id


def rank_summary(strategy_results: Dict[str, Dict]) -> List[Dict]:
    ranked = []
    for strategy_name, payload in strategy_results.items():
        metrics = payload["metrics"]
        ranked.append(
            {
                "strategy": strategy_name,
                "top1": metrics["top1"],
                "top5": metrics.get("top5", None),
            }
        )
    ranked.sort(key=lambda item: (-item["top1"], item["strategy"]))
    return ranked


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
    checkpoint_path, expected_task_id = checkpoint_spec(args, cli_args, data_manager)
    checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id)

    if not (hasattr(model._network, "backbone") and hasattr(model._network, "fc")):
        raise TypeError("This script currently supports TunaMax-style models with backbone + fc.")

    loader = build_test_loader(
        data_manager=data_manager,
        total_classes=model._total_classes,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
    )
    strategy_names = resolve_strategy_names(cli_args.strategies)
    expert_class_masks = build_expert_class_masks(data_manager, model._cur_task, model._total_classes)
    strategy_results = evaluate_fusion_strategies(
        model=model,
        loader=loader,
        strategy_names=strategy_names,
        expert_class_masks=expert_class_masks,
        topk_experts=cli_args.topk_experts,
        score_temperature=cli_args.score_temperature,
        entropy_temperature=cli_args.entropy_temperature,
        apply_scale=not cli_args.no_scale,
    )

    baseline_top1 = strategy_results["baseline_max_logit_route"]["metrics"]["top1"]
    for strategy_name, payload in strategy_results.items():
        payload["delta_vs_baseline_top1"] = round(payload["metrics"]["top1"] - baseline_top1, 4)

    return {
        "seed": seed,
        "model_name": args["model_name"],
        "dataset": args["dataset"],
        "checkpoint": str(checkpoint_path),
        "task_id": model._cur_task,
        "known_classes": model._known_classes,
        "total_classes": model._total_classes,
        "applied_scale": not cli_args.no_scale,
        "topk_experts": cli_args.topk_experts,
        "score_temperature": cli_args.score_temperature,
        "entropy_temperature": cli_args.entropy_temperature,
        "checkpoint_full_cnn": checkpoint.get("cnn_accy"),
        "strategies": strategy_results,
        "ranking": rank_summary(strategy_results),
    }


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [spie_v8_classifier_fusion_eval] %(message)s")
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
