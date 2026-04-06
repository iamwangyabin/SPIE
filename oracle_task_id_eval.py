import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch

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

    raw_devices = args.get("device", [])
    parsed_devices = []
    for device in raw_devices:
        if isinstance(device, torch.device):
            parsed_devices.append(device)
            continue

        device_text = str(device).strip().lower()
        if device_text.startswith("cuda:"):
            parsed_devices.append(torch.device(device_text))
        elif device_text.isdigit():
            parsed_devices.append(torch.device("cuda:{}".format(device_text)))

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


def run_single_seed(args):
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
    results = []

    for task_id in range(data_manager.nb_tasks):
        logging.info("Running task %s/%s", task_id + 1, data_manager.nb_tasks)
        model.incremental_train(data_manager)
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
                "full_cnn": full_cnn_accy,
                "oracle_cnn": oracle_cnn_accy,
            }
        )
        model.after_task()

    return results


def main():
    cli_args = setup_parser().parse_args()
    config = load_json(cli_args.config)

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
