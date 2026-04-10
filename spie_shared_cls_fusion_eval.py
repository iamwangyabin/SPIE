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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate post-hoc shared-CLS/expert fusion rules on a trained SPiE v9/v11-style checkpoint "
            "without any retraining."
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
        "--alpha-values",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated expert mixing weights for alpha*expert + (1-alpha)*shared.",
    )
    parser.add_argument(
        "--temperature-values",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated temperatures used by probability-based strategies.",
    )
    parser.add_argument(
        "--gate-temperature-values",
        type=str,
        default="0.5,1.0,2.0",
        help="Comma-separated temperatures for soft confidence gating.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply branch logits by args['scale'] before probability/gating computations.",
    )
    parser.add_argument("--output", type=str, default="", help="Optional path to save the summary JSON.")
    return parser


def parse_float_list(raw: str) -> List[float]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


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


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)


def margin_from_scores(scores: torch.Tensor) -> torch.Tensor:
    topk = min(2, scores.shape[1])
    top_values = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True).values
    if topk == 1:
        return top_values[:, 0]
    return top_values[:, 0] - top_values[:, 1]


def mix_name(prefix: str, alpha: float, temperature: Optional[float] = None) -> str:
    alpha_text = f"{alpha:.2f}"
    if temperature is None:
        return f"{prefix}_a{alpha_text}"
    return f"{prefix}_a{alpha_text}_t{temperature:.2f}"


def gate_name(prefix: str, temperature: float) -> str:
    return f"{prefix}_t{temperature:.2f}"


def branch_scores(
    model,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not all(
        hasattr(model, attr)
        for attr in ["_active_expert_ids", "_stack_calibrated_expert_logits", "_fuse_calibrated_expert_logits", "_shared_cls_logits"]
    ):
        raise TypeError("This script requires a SPiE v9/v11-style learner with shared CLS fusion helpers.")

    active_expert_ids = model._active_expert_ids()
    stacked_logits = model._stack_calibrated_expert_logits(inputs, active_expert_ids)
    expert_logits = model._fuse_calibrated_expert_logits(stacked_logits)
    shared_logits = model._shared_cls_logits(inputs)
    if shared_logits is None:
        raise RuntimeError("Checkpoint/model does not provide shared CLS logits.")
    return expert_logits, shared_logits


def build_strategy_scores(
    expert_logits: torch.Tensor,
    shared_logits: torch.Tensor,
    alpha_values: Sequence[float],
    temperature_values: Sequence[float],
    gate_temperature_values: Sequence[float],
    logit_scale: float,
    known_classes: int,
) -> Dict[str, torch.Tensor]:
    strategy_scores = {}

    strategy_scores["expert_only"] = expert_logits
    strategy_scores["shared_only"] = shared_logits
    strategy_scores["baseline_logit_sum_1_1"] = expert_logits + shared_logits

    old_mask = None
    new_mask = None
    if known_classes > 0:
        old_mask = torch.arange(expert_logits.shape[1], device=expert_logits.device) < known_classes
        new_mask = ~old_mask
        old_expert_new_shared = expert_logits.clone()
        old_expert_new_shared[:, new_mask] = shared_logits[:, new_mask]
        strategy_scores["classwise_old_expert_new_shared"] = old_expert_new_shared

        old_shared_new_expert = shared_logits.clone()
        old_shared_new_expert[:, new_mask] = expert_logits[:, new_mask]
        strategy_scores["classwise_old_shared_new_expert"] = old_shared_new_expert

    for alpha in alpha_values:
        strategy_scores[mix_name("logit_mix", alpha)] = alpha * expert_logits + (1.0 - alpha) * shared_logits

    scaled_expert = logit_scale * expert_logits
    scaled_shared = logit_scale * shared_logits

    for temperature in temperature_values:
        temp = max(float(temperature), 1e-8)
        expert_probs = F.softmax(scaled_expert / temp, dim=1)
        shared_probs = F.softmax(scaled_shared / temp, dim=1)

        expert_max_prob = expert_probs.max(dim=1).values
        shared_max_prob = shared_probs.max(dim=1).values
        expert_entropy = entropy_from_probs(expert_probs)
        shared_entropy = entropy_from_probs(shared_probs)
        expert_margin = margin_from_scores(expert_probs)
        shared_margin = margin_from_scores(shared_probs)

        for alpha in alpha_values:
            strategy_scores[mix_name("prob_mix", alpha, temp)] = alpha * expert_probs + (1.0 - alpha) * shared_probs

        choose_expert = expert_max_prob >= shared_max_prob
        strategy_scores[gate_name("select_maxprob", temp)] = torch.where(
            choose_expert.unsqueeze(1),
            expert_logits,
            shared_logits,
        )

        choose_expert = expert_entropy <= shared_entropy
        strategy_scores[gate_name("select_minentropy", temp)] = torch.where(
            choose_expert.unsqueeze(1),
            expert_logits,
            shared_logits,
        )

        choose_expert = expert_margin >= shared_margin
        strategy_scores[gate_name("select_margin", temp)] = torch.where(
            choose_expert.unsqueeze(1),
            expert_logits,
            shared_logits,
        )

        if old_mask is not None:
            old_expert_new_shared_prob = expert_probs.clone()
            old_expert_new_shared_prob[:, new_mask] = shared_probs[:, new_mask]
            strategy_scores[gate_name("classwise_old_expert_new_shared_prob", temp)] = old_expert_new_shared_prob

            old_shared_new_expert_prob = shared_probs.clone()
            old_shared_new_expert_prob[:, new_mask] = expert_probs[:, new_mask]
            strategy_scores[gate_name("classwise_old_shared_new_expert_prob", temp)] = old_shared_new_expert_prob

    for gate_temperature in gate_temperature_values:
        temp = max(float(gate_temperature), 1e-8)
        expert_probs = F.softmax(scaled_expert, dim=1)
        shared_probs = F.softmax(scaled_shared, dim=1)
        expert_conf = expert_probs.max(dim=1).values
        shared_conf = shared_probs.max(dim=1).values
        conf_logits = torch.stack((expert_conf, shared_conf), dim=1) / temp
        conf_weights = F.softmax(conf_logits, dim=1)
        strategy_scores[gate_name("soft_maxprob_gate", temp)] = (
            conf_weights[:, :1] * expert_probs + conf_weights[:, 1:] * shared_probs
        )

        expert_margin = margin_from_scores(expert_probs)
        shared_margin = margin_from_scores(shared_probs)
        margin_logits = torch.stack((expert_margin, shared_margin), dim=1) / temp
        margin_weights = F.softmax(margin_logits, dim=1)
        strategy_scores[gate_name("soft_margin_gate", temp)] = (
            margin_weights[:, :1] * expert_probs + margin_weights[:, 1:] * shared_probs
        )

    return strategy_scores


def evaluate_fusion_strategies(
    model,
    loader,
    alpha_values: Sequence[float],
    temperature_values: Sequence[float],
    gate_temperature_values: Sequence[float],
    apply_scale: bool,
) -> Dict[str, Dict]:
    model._network.eval()
    y_true = []
    prediction_bank: Dict[str, List[np.ndarray]] = {}
    logit_scale = float(model.args.get("scale", 1.0)) if apply_scale else 1.0

    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        with torch.no_grad():
            expert_logits, shared_logits = branch_scores(model, inputs)
            batch_scores = build_strategy_scores(
                expert_logits=expert_logits,
                shared_logits=shared_logits,
                alpha_values=alpha_values,
                temperature_values=temperature_values,
                gate_temperature_values=gate_temperature_values,
                logit_scale=logit_scale,
                known_classes=model._known_classes,
            )

        y_true.append(targets.numpy())
        for strategy_name, score_matrix in batch_scores.items():
            predicts = torch.topk(
                score_matrix,
                k=min(model.topk, score_matrix.shape[1]),
                dim=1,
                largest=True,
                sorted=True,
            )[1]
            predicts = finalize_predictions(predicts, model.topk)
            prediction_bank.setdefault(strategy_name, []).append(predicts.cpu().numpy())

    y_true = np.concatenate(y_true)
    results = {}
    for strategy_name, predictions in prediction_bank.items():
        y_pred = np.concatenate(predictions)
        results[strategy_name] = {
            "metrics": evaluate_predictions(model, y_pred, y_true),
        }
    return results


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


def summarize_branch_agreement(
    model,
    loader,
) -> Dict[str, float]:
    total = 0
    agree = 0
    expert_correct = 0
    shared_correct = 0

    model._network.eval()
    for _, inputs, targets in loader:
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        with torch.no_grad():
            expert_logits, shared_logits = branch_scores(model, inputs)

        expert_pred = expert_logits.argmax(dim=1)
        shared_pred = shared_logits.argmax(dim=1)
        agree += int(expert_pred.eq(shared_pred).sum().item())
        expert_correct += int(expert_pred.eq(targets).sum().item())
        shared_correct += int(shared_pred.eq(targets).sum().item())
        total += int(targets.numel())

    return {
        "agreement_rate": round(100.0 * agree / max(total, 1), 4),
        "expert_top1": round(100.0 * expert_correct / max(total, 1), 4),
        "shared_top1": round(100.0 * shared_correct / max(total, 1), 4),
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

    model = factory.get_model(args["model_name"], args)
    checkpoint_path, expected_task_id = checkpoint_spec(args, cli_args, data_manager)
    checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id)

    if not hasattr(model._network, "fc_shared_cls") or model._network.fc_shared_cls is None:
        raise TypeError("This script requires a checkpoint/model with fc_shared_cls.")

    loader = build_test_loader(
        data_manager=data_manager,
        total_classes=model._total_classes,
        batch_size=args["batch_size"],
        num_workers=cli_args.num_workers,
    )

    alpha_values = parse_float_list(cli_args.alpha_values)
    temperature_values = parse_float_list(cli_args.temperature_values)
    gate_temperature_values = parse_float_list(cli_args.gate_temperature_values)

    strategy_results = evaluate_fusion_strategies(
        model=model,
        loader=loader,
        alpha_values=alpha_values,
        temperature_values=temperature_values,
        gate_temperature_values=gate_temperature_values,
        apply_scale=not cli_args.no_scale,
    )

    baseline_top1 = strategy_results["baseline_logit_sum_1_1"]["metrics"]["top1"]
    for payload in strategy_results.values():
        payload["delta_vs_baseline_top1"] = round(payload["metrics"]["top1"] - baseline_top1, 4)

    return {
        "seed": seed,
        "model_name": args["model_name"],
        "dataset": args["dataset"],
        "checkpoint": str(checkpoint_path),
        "task_id": model._cur_task,
        "known_classes": model._known_classes,
        "total_classes": model._total_classes,
        "applied_scale_for_prob_strategies": not cli_args.no_scale,
        "alpha_values": alpha_values,
        "temperature_values": temperature_values,
        "gate_temperature_values": gate_temperature_values,
        "checkpoint_full_cnn": checkpoint.get("cnn_accy"),
        "branch_summary": summarize_branch_agreement(model, loader),
        "strategies": strategy_results,
        "ranking": rank_summary(strategy_results),
    }


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [spie_shared_cls_fusion_eval] %(message)s")
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
