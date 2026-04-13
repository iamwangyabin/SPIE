import argparse
import copy
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import accuracy


def load_json(setting_path):
    with open(setting_path) as data_file:
        return json.load(data_file)


def setup_parser():
    parser = argparse.ArgumentParser(description="Standalone oracle task-id evaluation for SPiE v2 / TunaMax-style models.")
    parser.add_argument("--config", type=str, required=True, help="Json config file.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save oracle/full evaluation results as JSON.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory containing task_0.pkl, task_1.pkl, ... checkpoints saved by trainer.py.",
    )
    return parser


def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(args):
    if not torch.cuda.is_available():
        args["device"] = [torch.device("cpu")]
        return

    cuda_count = torch.cuda.device_count()
    raw_devices = args.get("device", [])
    parsed_devices = []
    for device in raw_devices:
        if isinstance(device, torch.device):
            parsed_device = device
        else:
            device_text = str(device).strip().lower()
            if device_text.startswith("cuda:"):
                parsed_device = torch.device(device_text)
            elif device_text.isdigit():
                parsed_device = torch.device("cuda:{}".format(device_text))
            else:
                continue

        if parsed_device.type != "cuda":
            parsed_devices.append(parsed_device)
            continue

        if parsed_device.index is None:
            parsed_devices.append(torch.device("cuda:0"))
            continue

        if 0 <= parsed_device.index < cuda_count:
            parsed_devices.append(parsed_device)
            continue

        logging.warning(
            "Ignoring unavailable CUDA device %s. PyTorch sees %s CUDA device(s); "
            "CUDA_VISIBLE_DEVICES=%s. Use visible ordinals such as cuda:0 after CUDA_VISIBLE_DEVICES remapping.",
            parsed_device,
            cuda_count,
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        )

    args["device"] = parsed_devices or [torch.device("cuda:0")]


def evaluate_predictions(model, y_pred, y_true):
    grouped = accuracy(
        y_pred[:, 0],
        y_true,
        model._known_classes,
        model.args["init_cls"],
        model.args["increment"],
    )
    result = {
        "grouped": grouped,
        "top1": grouped["total"],
        "top{}".format(model.topk): np.around(
            (y_pred.T == np.tile(y_true, (model.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        ).item(),
    }
    return result


def get_task_class_indices(model, task_id):
    indices = [class_idx for class_idx, mapped_task in model.cls2task.items() if mapped_task == task_id]
    indices = [class_idx for class_idx in indices if class_idx < model._total_classes]
    if not indices:
        raise ValueError("No classes found for task {}.".format(task_id))
    return sorted(indices)


def eval_oracle_task_id(model, loader):
    model._network.eval()
    y_pred, y_true = [], []

    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        batch_predicts = torch.full(
            (targets.shape[0], model.topk),
            -1,
            device=targets.device,
            dtype=torch.long,
        )

        with torch.no_grad():
            target_task_ids = torch.tensor(
                [model.cls2task[int(target.item())] for target in targets],
                device=targets.device,
                dtype=torch.long,
            )

            for task_id in target_task_ids.unique(sorted=True).tolist():
                sample_mask = target_task_ids == task_id
                task_inputs = inputs[sample_mask]
                task_features = model._network.backbone(task_inputs, adapter_id=task_id, train=False)["features"]
                task_logits = model._network.fc(task_features)["logits"][:, : model._total_classes] * model.args["scale"]

                class_indices = torch.tensor(
                    get_task_class_indices(model, task_id),
                    device=task_logits.device,
                    dtype=torch.long,
                )
                task_logits = task_logits.index_select(dim=1, index=class_indices)

                topk = min(model.topk, task_logits.shape[1])
                task_predicts = torch.topk(task_logits, k=topk, dim=1, largest=True, sorted=True)[1]
                task_predicts = class_indices[task_predicts]

                if topk < model.topk:
                    pad = torch.full(
                        (task_predicts.shape[0], model.topk - topk),
                        -1,
                        device=task_predicts.device,
                        dtype=task_predicts.dtype,
                    )
                    task_predicts = torch.cat([task_predicts, pad], dim=1)

                batch_predicts[sample_mask] = task_predicts

        y_pred.append(batch_predicts.cpu().numpy())
        y_true.append(targets.cpu().numpy())

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return evaluate_predictions(model, y_pred, y_true)


def resolve_checkpoint_dir(args):
    checkpoint_dir = args.get("checkpoint_dir", "")
    if not checkpoint_dir:
        raise ValueError("oracle_task_id_eval.py requires --checkpoint-dir or checkpoint_dir in the config.")
    return Path(str(checkpoint_dir).format(seed=args["seed"]))


def total_classes_after_task(data_manager, task_id):
    return sum(data_manager.get_task_size(task_idx) for task_idx in range(task_id + 1))


def rebuild_task_metadata(model, data_manager, task_id, total_classes):
    model._cur_task = task_id
    model._total_classes = total_classes
    model._known_classes = total_classes - data_manager.get_task_size(task_id)
    model.cls2task = {}

    class_offset = 0
    for mapped_task_id in range(task_id + 1):
        task_size = data_manager.get_task_size(mapped_task_id)
        for class_idx in range(class_offset, class_offset + task_size):
            model.cls2task[class_idx] = mapped_task_id
        class_offset += task_size


def prepare_checkpoint_structure(model, data_manager, task_id, state_dict):
    backbone = model._backbone_module() if hasattr(model, "_backbone_module") else model._network.backbone

    current_heads = 0 if model._network.fc is None else len(model._network.fc.heads)
    for mapped_task_id in range(current_heads, task_id + 1):
        model._network.update_fc(data_manager.get_task_size(mapped_task_id))

    if hasattr(model, "_prepare_task_modules_for_load"):
        model._prepare_task_modules_for_load(task_id, data_manager, state_dict)

    current_adapters = len(backbone.adapter_list) if hasattr(backbone, "adapter_list") else task_id + 1
    for _ in range(current_adapters, task_id + 1):
        if hasattr(backbone, "adapter_update"):
            backbone.adapter_update()

    has_merged_adapter = any(key.startswith("backbone.merged_adapter.") for key in state_dict)
    if has_merged_adapter and hasattr(backbone, "merged_adapter") and len(backbone.merged_adapter) == 0:
        backbone.merged_adapter = copy.deepcopy(backbone.cur_adapter)


def load_task_checkpoint(model, data_manager, checkpoint_path, expected_task_id):
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task_id = int(checkpoint["task_id"] if "task_id" in checkpoint else checkpoint["tasks"])
    if task_id != expected_task_id:
        raise ValueError(
            "Checkpoint {} contains task_id={}, but task_{}.pkl was expected.".format(
                checkpoint_path,
                task_id,
                expected_task_id,
            )
        )

    total_classes = int(checkpoint["total_classes"])
    expected_total_classes = total_classes_after_task(data_manager, task_id)
    if total_classes != expected_total_classes:
        raise ValueError(
            "Checkpoint {} has total_classes={}, but dataset/config expects {} for task {}.".format(
                checkpoint_path,
                total_classes,
                expected_total_classes,
                task_id,
            )
        )

    model._network.to("cpu")
    state_dict = checkpoint["model_state_dict"]
    prepare_checkpoint_structure(model, data_manager, task_id, state_dict)
    incompatible = model._network.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint {} did not match the model. Missing keys: {}. Unexpected keys: {}.".format(
                checkpoint_path,
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        )

    rebuild_task_metadata(model, data_manager, task_id, total_classes)
    model._network.to(model._device)
    model._network.eval()
    return checkpoint


def build_test_loader(data_manager, model, total_classes):
    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    return DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False, num_workers=8)


def run_single_seed(args):
    set_random(args["seed"])
    set_device(args)
    checkpoint_dir = resolve_checkpoint_dir(args)

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
    results = []

    for task_id in range(data_manager.nb_tasks):
        checkpoint_path = checkpoint_dir / "task_{}.pkl".format(task_id)
        logging.info("Evaluating checkpoint %s", checkpoint_path)
        checkpoint = load_task_checkpoint(model, data_manager, checkpoint_path, task_id)
        model.test_loader = build_test_loader(data_manager, model, model._total_classes)

        full_cnn_accy, _ = model.eval_task()
        oracle_cnn_accy = eval_oracle_task_id(model, model.test_loader)

        logging.info("Task %s full-CI top1: %.2f", task_id, full_cnn_accy["top1"])
        logging.info("Task %s oracle top1: %.2f", task_id, oracle_cnn_accy["top1"])
        logging.info("Task %s full-CI grouped: %s", task_id, full_cnn_accy["grouped"])
        logging.info("Task %s oracle grouped: %s", task_id, oracle_cnn_accy["grouped"])

        results.append(
            {
                "task_id": task_id,
                "known_classes": model._known_classes,
                "total_classes": model._total_classes,
                "checkpoint": str(checkpoint_path),
                "full_cnn": full_cnn_accy,
                "oracle_cnn": oracle_cnn_accy,
                "checkpoint_full_cnn": checkpoint.get("cnn_accy"),
            }
        )

    return results


def main():
    cli_args = setup_parser().parse_args()
    config = load_json(cli_args.config)
    if cli_args.checkpoint_dir:
        config["checkpoint_dir"] = cli_args.checkpoint_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [oracle_task_id_eval] %(message)s",
    )

    all_results = []
    for seed in copy.deepcopy(config["seed"]):
        args = copy.deepcopy(config)
        args["seed"] = seed
        seed_results = run_single_seed(args)
        all_results.append({"seed": seed, "results": seed_results})

    if cli_args.output:
        output_path = Path(cli_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        logging.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()
