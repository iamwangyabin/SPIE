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
        description="Export shared-FC mistakes for SPiE v14 with shared logits and all expert OOD outputs."
    )
    parser.add_argument("--config", type=str, required=True, help="Experiment config JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path like logs/.../task_8.pkl.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of misclassified samples to save.")
    parser.add_argument(
        "--shared-topk",
        type=int,
        default=5,
        help="How many shared-FC top classes to export per sample.",
    )
    parser.add_argument("--note", type=str, default="shared-fc-mistakes", help="Optional logging note.")
    return parser


def _rebuild_spie_v14_from_checkpoint(model, data_manager, checkpoint):
    if model.args["model_name"] != "spie_v14":
        raise ValueError("This exporter currently only supports model_name='spie_v14'.")

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
        model.task_class_ranges.append((start, end))
        backbone.reset_task_modules()
        backbone.adapter_update()
        start = end

    _materialize_detector_buffers_from_checkpoint(model, checkpoint["model_state_dict"])
    missing, unexpected = model._network.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint structure mismatch. Missing={missing}, unexpected={unexpected}")

    model._network.to(model._device)
    model._network.eval()
    return task_id


def _materialize_detector_buffers_from_checkpoint(model, state_dict):
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


def _class_to_task_id(task_ranges, class_idx):
    for task_id, (start, end) in enumerate(task_ranges):
        if start <= class_idx < end:
            return task_id
    raise ValueError(f"Class {class_idx} is not covered by any task range.")


def _topk_payload(logits_row, k):
    topk = min(int(k), int(logits_row.shape[0]))
    values, indices = torch.topk(logits_row, k=topk, dim=0, largest=True, sorted=True)
    probs = torch.softmax(logits_row, dim=0)
    payload = []
    for rank in range(topk):
        class_id = int(indices[rank].item())
        payload.append(
            {
                "rank": rank + 1,
                "class_id": class_id,
                "task_id": None,
                "logit": round(float(values[rank].item()), 6),
                "prob": round(float(probs[class_id].item()), 6),
            }
        )
    return payload


def _round_list(values, ndigits=6):
    return [round(float(v), ndigits) for v in values]


def _to_jsonable_image_ref(image_ref):
    if isinstance(image_ref, str):
        return image_ref
    if isinstance(image_ref, np.generic):
        return image_ref.item()
    return image_ref


@torch.no_grad()
def export_shared_fc_mistakes(model, data_manager, loader, max_samples, shared_topk):
    active_expert_ids = model._active_expert_ids()
    dataset = loader.dataset
    results = []
    total_seen = 0
    total_wrong = 0

    for _, inputs, targets in loader:
        batch_dataset_indices = _.tolist() if torch.is_tensor(_) else list(_)
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)

        shared_logits = model._shared_cls_logits(inputs)
        shared_preds = torch.argmax(shared_logits, dim=1)

        ood_out = model._network.forward_multi_expert_ood_scores(inputs, active_expert_ids)
        task_weights = model._compute_task_prior_weights(ood_out["scores"])

        batch_size = targets.shape[0]
        total_seen += int(batch_size)

        for sample_offset in range(batch_size):
            target = int(targets[sample_offset].item())
            shared_pred = int(shared_preds[sample_offset].item())
            if shared_pred == target:
                continue

            total_wrong += 1
            if len(results) >= max_samples:
                continue

            dataset_index = int(batch_dataset_indices[sample_offset])
            image_ref = _to_jsonable_image_ref(dataset.images[dataset_index])
            sample_task_id = _class_to_task_id(model.task_class_ranges, target)

            expert_payload = []
            per_sample_scores = ood_out["scores"][:, sample_offset]
            per_sample_weights = task_weights[:, sample_offset]
            for local_idx, expert_id in enumerate(active_expert_ids):
                expert_payload.append(
                    {
                        "expert_id": int(expert_id),
                        "ood_score": round(float(per_sample_scores[local_idx].item()), 6),
                        "task_weight": round(float(per_sample_weights[local_idx].item()), 6),
                    }
                )
            results.append(
                {
                    "dataset_index": dataset_index,
                    "image": image_ref if isinstance(image_ref, str) else int(image_ref),
                    "target_class": target,
                    "target_task_id": int(sample_task_id),
                    "shared_pred_class": shared_pred,
                    "shared_logits": _round_list(shared_logits[sample_offset].detach().cpu().tolist()),
                    "expert_outputs": expert_payload,
                }
            )

    return {
        "num_test_samples": total_seen,
        "num_shared_fc_mistakes": total_wrong,
        "exported_mistakes": len(results),
        "samples": results,
    }


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

    model = factory.get_model(args["model_name"], args)
    task_id = _rebuild_spie_v14_from_checkpoint(model, data_manager, checkpoint)

    total_classes = int(checkpoint["total_classes"])
    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=cli_args.num_workers,
    )

    export_payload = export_shared_fc_mistakes(
        model=model,
        data_manager=data_manager,
        loader=test_loader,
        max_samples=max(cli_args.max_samples, 0),
        shared_topk=max(cli_args.shared_topk, 1),
    )
    export_payload.update(
        {
            "checkpoint": cli_args.checkpoint,
            "task_id": task_id,
            "total_classes": total_classes,
            "max_samples": int(cli_args.max_samples),
            "shared_topk": int(cli_args.shared_topk),
        }
    )

    output_dir = os.path.dirname(os.path.abspath(cli_args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(cli_args.output, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    logging.info(
        "Exported %s / %s shared-FC mistakes to %s",
        export_payload["exported_mistakes"],
        export_payload["num_shared_fc_mistakes"],
        cli_args.output,
    )


if __name__ == "__main__":
    main()
