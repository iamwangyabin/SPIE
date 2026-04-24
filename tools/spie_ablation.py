import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running this script directly from tools/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import factory
from utils.data_manager import DataManager


def set_random(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-12, None)


def clean_marginal_fusion_probs_np(model, shared_logits, expert_logits_by_task):
    """
    Clean posterior fusion:

        p_final(c | x)
        =
        p_shared(y in task(c) | x)
        *
        p_expert(c | x, y in task(c))

    No alpha, no temperature, no extra task-softmax.
    """
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_clean = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start_idx, end_idx) in enumerate(model.task_class_ranges):
        task_mass = p_shared[:, start_idx:end_idx].sum(axis=1, keepdims=True)
        local_prob = stable_softmax_np(expert_logits_by_task[task_id]).astype(np.float32)
        p_clean[:, start_idx:end_idx] = task_mass * local_prob

    return p_clean


def product_marginal_fusion_probs_np(model, shared_logits, expert_logits_by_task):
    p_shared = stable_softmax_np(shared_logits).astype(np.float32)
    p_final = np.zeros_like(p_shared, dtype=np.float32)

    for task_id, (start, end) in enumerate(model.task_class_ranges):
        block = p_shared[:, start:end]
        task_mass = block.sum(axis=1, keepdims=True)
        shared_local = block / np.clip(task_mass, 1e-12, None)
        expert_local = stable_softmax_np(expert_logits_by_task[task_id]).astype(np.float32)

        local = shared_local * expert_local
        local = local / np.clip(local.sum(axis=1, keepdims=True), 1e-12, None)

        p_final[:, start:end] = task_mass * local

    return p_final


def oracle_task_local_acc(model, shared_logits_np, expert_logits_by_task, y_true):
    shared_correct = 0
    expert_correct = 0
    total = len(y_true)

    for i, y in enumerate(y_true):
        true_task = None
        local_y = None
        for task_id, (start, end) in enumerate(model.task_class_ranges):
            if start <= y < end:
                true_task = task_id
                local_y = y - start
                break

        if true_task is None:
            raise ValueError(f"Target {int(y)} does not belong to any task class range.")

        start, end = model.task_class_ranges[true_task]
        shared_local_pred = np.argmax(shared_logits_np[i, start:end])
        expert_local_pred = np.argmax(expert_logits_by_task[true_task][i])

        if shared_local_pred == local_y:
            shared_correct += 1
        if expert_local_pred == local_y:
            expert_correct += 1

    return {
        "shared_oracle_task_local_top1": 100.0 * shared_correct / total,
        "expert_oracle_task_local_top1": 100.0 * expert_correct / total,
    }


def build_model_structure_without_training(model, data_manager, num_tasks):
    """
    Run incremental_train only to initialize task structure, heads, and loaders.
    model._train is temporarily replaced with a no-op so no training is run.
    """
    original_train = model._train

    def noop_train(train_loader):
        return None

    model._train = noop_train
    try:
        for _ in range(num_tasks):
            model.incremental_train(data_manager)

            backbone = (
                model._backbone_module()
                if hasattr(model, "_backbone_module")
                else model._network.backbone
            )
            if hasattr(backbone, "adapter_update"):
                backbone.adapter_update()

            model.after_task()
    finally:
        model._train = original_train


def set_eval_known_classes(model, ckpt):
    """
    Checkpoints are saved after after_task(), so known_classes often equals
    total_classes. For grouped old/new accuracy, use the current task start.
    """
    ckpt_known_classes = int(ckpt["known_classes"])
    total_classes = int(ckpt["total_classes"])
    if ckpt_known_classes >= total_classes and model.task_class_ranges:
        model._known_classes = int(model.task_class_ranges[-1][0])
    else:
        model._known_classes = ckpt_known_classes


def print_metric(name, accy):
    print(f"\n[{name}]")
    print(f"top1: {accy['top1']:.2f}")
    print(f"top5: {accy['top5']:.2f}")
    print("grouped:", accy["grouped"])


def parse_device(device_arg: str):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_arg}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SPiE checkpoint ablations: shared FC, original posterior fusion, and clean marginal fusion."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the SPiE experiment json.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved task_*.pkl checkpoint.")
    parser.add_argument("--device", type=str, default="0", help="CUDA id, or 'cpu'. Default: 0.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save top1 summary.")
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        args = json.load(f)

    if args.get("model_name", "").lower() != "spie":
        raise ValueError(f"This script only supports model_name='spie', got {args.get('model_name')!r}.")

    args["swanlab"] = False
    args["spie_backbone_dataparallel"] = False

    if isinstance(args.get("seed"), list):
        args["seed"] = args["seed"][0]
    if args_cli.seed is not None:
        args["seed"] = args_cli.seed

    device = parse_device(args_cli.device)
    args["device"] = [device]

    set_random(int(args["seed"]))

    ckpt = torch.load(args_cli.checkpoint, map_location="cpu")
    target_task = int(ckpt["tasks"])
    num_tasks = target_task + 1

    print(f"Loaded checkpoint: {args_cli.checkpoint}")
    print(f"Checkpoint task: {target_task}")
    print(f"Checkpoint total_classes: {ckpt['total_classes']}")

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

    if num_tasks > data_manager.nb_tasks:
        raise ValueError(
            f"Checkpoint asks for {num_tasks} tasks, but config/data define only {data_manager.nb_tasks} tasks."
        )

    model = factory.get_model(args["model_name"], args)

    build_model_structure_without_training(model, data_manager, num_tasks)

    model._cur_task = target_task
    model._total_classes = int(ckpt["total_classes"])
    set_eval_known_classes(model, ckpt)

    missing, unexpected = model._network.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    if missing:
        print("\n[Warning] Missing keys:")
        for key in missing[:20]:
            print("  ", key)
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if unexpected:
        print("\n[Warning] Unexpected keys:")
        for key in unexpected[:20]:
            print("  ", key)
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    model._network.to(device)
    model._network.eval()

    shared_logits_np, expert_logits_by_task, y_true = model._collect_eval_logits_np(
        model.test_loader
    )

    shared_pred = model._predict_topk_np(shared_logits_np)
    shared_accy = model._evaluate(shared_pred, y_true)

    p_moe, p_final = model._posterior_fusion_probs_np(
        shared_logits_np,
        expert_logits_by_task,
    )
    p_moe_pred = model._predict_topk_np(p_moe)
    p_final_pred = model._predict_topk_np(p_final)

    p_moe_accy = model._evaluate(p_moe_pred, y_true)
    p_final_accy = model._evaluate(p_final_pred, y_true)

    p_clean = clean_marginal_fusion_probs_np(
        model,
        shared_logits_np,
        expert_logits_by_task,
    )
    p_clean_pred = model._predict_topk_np(p_clean)
    p_clean_accy = model._evaluate(p_clean_pred, y_true)

    p_product = product_marginal_fusion_probs_np(
        model,
        shared_logits_np,
        expert_logits_by_task,
    )
    p_product_pred = model._predict_topk_np(p_product)
    p_product_accy = model._evaluate(p_product_pred, y_true)

    oracle_local_accy = oracle_task_local_acc(
        model,
        shared_logits_np,
        expert_logits_by_task,
        y_true,
    )

    print_metric("shared_fc", shared_accy)
    print_metric("original_p_moe", p_moe_accy)
    print_metric("original_p_final", p_final_accy)
    print_metric("clean_marginal_fusion", p_clean_accy)
    print_metric("product_marginal_fusion", p_product_accy)

    print("\n[oracle_task_local]")
    for key, value in oracle_local_accy.items():
        print(f"{key}: {value:.2f}")

    all_results = {
        "shared_fc": shared_accy["top1"],
        "original_p_moe": p_moe_accy["top1"],
        "original_p_final": p_final_accy["top1"],
        "clean_marginal_fusion": p_clean_accy["top1"],
        "product_marginal_fusion": p_product_accy["top1"],
    }
    best_name = max(all_results, key=all_results.get)

    print("\n========== Summary ==========")
    for key, value in all_results.items():
        print(f"{key:24s}: {value:.2f}")
    print(f"\nBest top1: {best_name} = {all_results[best_name]:.2f}")

    row_sum = p_clean.sum(axis=1)
    print(
        "\nClean fusion probability sum:",
        f"min={row_sum.min():.6f}",
        f"max={row_sum.max():.6f}",
        f"mean={row_sum.mean():.6f}",
    )

    product_row_sum = p_product.sum(axis=1)
    print(
        "Product fusion probability sum:",
        f"min={product_row_sum.min():.6f}",
        f"max={product_row_sum.max():.6f}",
        f"mean={product_row_sum.mean():.6f}",
    )

    if args_cli.output_json:
        output = {
            "checkpoint": args_cli.checkpoint,
            "task": target_task,
            "total_classes": int(ckpt["total_classes"]),
            "eval_known_classes": int(model._known_classes),
            "top1": all_results,
            "oracle_task_local": oracle_local_accy,
            "best_top1": {"name": best_name, "value": all_results[best_name]},
            "clean_probability_sum": {
                "min": float(row_sum.min()),
                "max": float(row_sum.max()),
                "mean": float(row_sum.mean()),
            },
            "product_probability_sum": {
                "min": float(product_row_sum.min()),
                "max": float(product_row_sum.max()),
                "mean": float(product_row_sum.mean()),
            },
        }
        output_path = Path(args_cli.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved summary json: {output_path}")


if __name__ == "__main__":
    main()
