import argparse
import copy
import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import factory
from utils.data_manager import DataManager


SUBSETS = ("all", "in_task", "out_task")
STAT_NAMES = ("mean", "std", "median", "p90", "p95", "p99")


def load_json(setting_path: str) -> Dict:
    with open(setting_path) as data_file:
        return json.load(data_file)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-expert maximum response distributions on the test set "
            "for incremental-learning checkpoints."
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
        help="Task checkpoint to analyze: latest, final, or an integer task id. Default: latest.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for CSV/JSON/plot outputs. Defaults to <checkpoint_dir>/expert_response_analysis/task_<id>.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Analyze only this seed. Defaults to all config seeds.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size for analysis.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for the test loader.")
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not multiply TunaLinear/TunaMax-style cosine scores by args['scale'].",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Only write CSV/JSON outputs.")
    return parser


def set_random(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(args: Dict) -> None:
    if not torch.cuda.is_available():
        args["device"] = [torch.device("cpu")]
        return

    cuda_count = torch.cuda.device_count()
    parsed_devices = []
    for device in args.get("device", []):
        if isinstance(device, torch.device):
            parsed_device = device
        else:
            device_text = str(device).strip().lower()
            if device_text.startswith("cuda:"):
                parsed_device = torch.device(device_text)
            elif device_text.isdigit():
                parsed_device = torch.device(f"cuda:{device_text}")
            else:
                continue

        if parsed_device.type != "cuda":
            parsed_devices.append(parsed_device)
        elif parsed_device.index is None:
            parsed_devices.append(torch.device("cuda:0"))
        elif 0 <= parsed_device.index < cuda_count:
            parsed_devices.append(parsed_device)
        else:
            logging.warning("Ignoring unavailable CUDA device %s.", parsed_device)

    args["device"] = parsed_devices or [torch.device("cuda:0")]


def iter_config_seeds(config: Dict, selected_seed: Optional[int]) -> List[int]:
    if selected_seed is not None:
        return [selected_seed]
    raw_seed = config.get("seed", [0])
    if isinstance(raw_seed, (list, tuple)):
        return [int(seed) for seed in raw_seed]
    return [int(raw_seed)]


def resolve_checkpoint_dir(args: Dict) -> Path:
    checkpoint_dir = args.get("checkpoint_dir", "")
    if not checkpoint_dir:
        raise ValueError("Set --checkpoint-dir or checkpoint_dir in the config.")
    return Path(str(checkpoint_dir).format(seed=args["seed"]))


def find_latest_task_id(checkpoint_dir: Path) -> int:
    task_ids = []
    for checkpoint_path in checkpoint_dir.glob("task_*.pkl"):
        match = re.fullmatch(r"task_(\d+)\.pkl", checkpoint_path.name)
        if match is not None:
            task_ids.append(int(match.group(1)))
    if not task_ids:
        raise FileNotFoundError(f"No task_*.pkl checkpoints found in {checkpoint_dir}.")
    return max(task_ids)


def resolve_task_id(task_id_arg: str, checkpoint_dir: Optional[Path], data_manager: DataManager) -> int:
    value = str(task_id_arg).strip().lower()
    if value in {"final", "last"}:
        return data_manager.nb_tasks - 1
    if value == "latest":
        if checkpoint_dir is None:
            return data_manager.nb_tasks - 1
        return find_latest_task_id(checkpoint_dir)
    return int(value)


def total_classes_after_task(data_manager: DataManager, task_id: int) -> int:
    return sum(data_manager.get_task_size(task_idx) for task_idx in range(task_id + 1))


def build_task_metadata(data_manager: DataManager, task_id: int, total_classes: int) -> Tuple[Dict[int, int], List[int]]:
    class_to_task: Dict[int, int] = {}
    task_offsets = []
    class_offset = 0
    for mapped_task_id in range(task_id + 1):
        task_offsets.append(class_offset)
        task_size = data_manager.get_task_size(mapped_task_id)
        for class_idx in range(class_offset, min(class_offset + task_size, total_classes)):
            class_to_task[class_idx] = mapped_task_id
        class_offset += task_size
    return class_to_task, task_offsets


def rebuild_model_task_state(model, data_manager: DataManager, task_id: int, total_classes: int) -> None:
    model._cur_task = task_id
    model._total_classes = total_classes
    model._known_classes = total_classes - data_manager.get_task_size(task_id)
    if hasattr(model, "cls2task"):
        class_to_task, _ = build_task_metadata(data_manager, task_id, total_classes)
        model.cls2task = class_to_task


def get_backbone(model):
    if hasattr(model, "_backbone_module"):
        return model._backbone_module()
    return getattr(model._network, "backbone", None)


def prepare_tunamax_style_modules(model, data_manager: DataManager, task_id: int, state_dict: Dict) -> None:
    if hasattr(model._network, "fc"):
        fc = model._network.fc
        if fc is None:
            current_heads = 0
        elif hasattr(fc, "heads"):
            current_heads = len(fc.heads)
        else:
            current_heads = 1

        for mapped_task_id in range(current_heads, task_id + 1):
            model._network.update_fc(data_manager.get_task_size(mapped_task_id))

    if hasattr(model, "_prepare_task_modules_for_load"):
        model._prepare_task_modules_for_load(task_id, data_manager, state_dict)

    backbone = get_backbone(model)
    if backbone is None:
        return

    if hasattr(backbone, "adapter_list"):
        current_adapters = len(backbone.adapter_list)
        for _ in range(current_adapters, task_id + 1):
            if hasattr(backbone, "adapter_update"):
                backbone.adapter_update()

    has_merged_adapter = any(key.startswith("backbone.merged_adapter.") for key in state_dict)
    if has_merged_adapter and hasattr(backbone, "merged_adapter") and len(backbone.merged_adapter) == 0:
        backbone.merged_adapter = copy.deepcopy(backbone.cur_adapter)


def load_task_checkpoint(model, data_manager: DataManager, checkpoint_path: Path, expected_task_id: Optional[int]) -> Dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task_id = int(checkpoint["task_id"] if "task_id" in checkpoint else checkpoint["tasks"])
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError(
            f"{checkpoint_path} contains task_id={task_id}, but task_{expected_task_id}.pkl was expected."
        )

    total_classes = int(checkpoint["total_classes"])
    expected_total_classes = total_classes_after_task(data_manager, task_id)
    if total_classes != expected_total_classes:
        raise ValueError(
            f"{checkpoint_path} has total_classes={total_classes}, but the dataset/config expects "
            f"{expected_total_classes} for task {task_id}."
        )

    model._network.to("cpu")
    state_dict = checkpoint["model_state_dict"]
    prepare_tunamax_style_modules(model, data_manager, task_id, state_dict)

    incompatible = model._network.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{checkpoint_path} did not match the model. "
            f"Missing keys: {incompatible.missing_keys}. Unexpected keys: {incompatible.unexpected_keys}."
        )

    rebuild_model_task_state(model, data_manager, task_id, total_classes)
    model._network.to(model._device)
    model._network.eval()
    return checkpoint


def build_test_loader(data_manager: DataManager, total_classes: int, batch_size: int, num_workers: int) -> DataLoader:
    test_dataset = data_manager.get_dataset(np.arange(0, total_classes), source="test", mode="test")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def is_incremental_expert_model(network) -> bool:
    return (
        hasattr(network, "heads")
        and hasattr(network, "backbone")
        and hasattr(network, "num_classes_per_expert")
        and not hasattr(network, "fc")
    )


@torch.no_grad()
def get_logits_for_expert(
    model,
    inputs: torch.Tensor,
    expert_id: int,
    total_classes: int,
    task_offsets: Sequence[int],
    apply_scale: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    network = model._network

    if is_incremental_expert_model(network):
        out = network.backbone(inputs, active_experts=[expert_id], return_dict=True)
        logits = network.heads.heads[expert_id](out.expert_pooled[:, 0, :])
        offset = task_offsets[expert_id]
        class_lookup = torch.arange(offset, offset + logits.shape[1], device=logits.device, dtype=torch.long)
        return logits, class_lookup

    if hasattr(network, "forward_single_expert"):
        logits = network.forward_single_expert(inputs, expert_idx=expert_id)["logits"][:, :total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        class_lookup = torch.arange(total_classes, device=logits.device, dtype=torch.long)
        return logits, class_lookup

    if hasattr(network, "backbone") and hasattr(network, "fc"):
        features = network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
        logits = network.fc(features)["logits"][:, :total_classes]
        if apply_scale:
            logits = logits * float(model.args.get("scale", 1.0))
        class_lookup = torch.arange(total_classes, device=logits.device, dtype=torch.long)
        return logits, class_lookup

    raise TypeError(f"Unsupported network type for expert response analysis: {type(network).__name__}")


def append_batch_records(
    rows: List[Dict],
    scores_by_expert: Dict[int, Dict[str, List[float]]],
    sample_ids: torch.Tensor,
    sample_offset_start: int,
    targets: torch.Tensor,
    class_to_task: Dict[int, int],
    expert_id: int,
    logits: torch.Tensor,
    class_lookup: torch.Tensor,
) -> None:
    topk = min(2, logits.shape[1])
    top_values, top_indices = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)
    top_classes = class_lookup[top_indices]

    top_values_cpu = top_values.detach().cpu().numpy()
    top_classes_cpu = top_classes.detach().cpu().numpy()
    sample_ids_cpu = sample_ids.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()

    for batch_idx, true_class_raw in enumerate(targets_cpu):
        true_class = int(true_class_raw)
        true_task = int(class_to_task[true_class])
        is_in_task = true_task == expert_id
        max_score = float(top_values_cpu[batch_idx, 0])
        top2_score = float(top_values_cpu[batch_idx, 1]) if topk > 1 else float("nan")
        top2_class = int(top_classes_cpu[batch_idx, 1]) if topk > 1 else -1

        rows.append(
            {
                "sample_offset": sample_offset_start + batch_idx,
                "sample_index": int(sample_ids_cpu[batch_idx]),
                "true_class": true_class,
                "true_task": true_task,
                "expert_id": int(expert_id),
                "is_in_task": int(is_in_task),
                "max_score": max_score,
                "argmax_class": int(top_classes_cpu[batch_idx, 0]),
                "top1_score": max_score,
                "top2_score": top2_score,
                "top2_class": top2_class,
            }
        )

        scores_by_expert[expert_id]["all"].append(max_score)
        if is_in_task:
            scores_by_expert[expert_id]["in_task"].append(max_score)
        else:
            scores_by_expert[expert_id]["out_task"].append(max_score)


def collect_expert_responses(
    model,
    loader: DataLoader,
    class_to_task: Dict[int, int],
    task_offsets: Sequence[int],
    apply_scale: bool,
) -> Tuple[List[Dict], Dict[int, Dict[str, List[float]]]]:
    model._network.eval()
    expert_ids = list(range(model._cur_task + 1))
    scores_by_expert = {expert_id: {subset: [] for subset in SUBSETS} for expert_id in expert_ids}
    rows: List[Dict] = []
    sample_offset = 0

    for _, (sample_ids, inputs, targets) in enumerate(loader):
        inputs = inputs.to(model._device)
        targets = targets.to(model._device)
        for expert_id in expert_ids:
            logits, class_lookup = get_logits_for_expert(
                model=model,
                inputs=inputs,
                expert_id=expert_id,
                total_classes=model._total_classes,
                task_offsets=task_offsets,
                apply_scale=apply_scale,
            )
            append_batch_records(
                rows=rows,
                scores_by_expert=scores_by_expert,
                sample_ids=sample_ids,
                sample_offset_start=sample_offset,
                targets=targets,
                class_to_task=class_to_task,
                expert_id=expert_id,
                logits=logits,
                class_lookup=class_lookup,
            )
        sample_offset += targets.shape[0]

    return rows, scores_by_expert


def summarize_array(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        stats = {name: float("nan") for name in STAT_NAMES}
        stats["count"] = 0
        return stats
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def build_summary(scores_by_expert: Dict[int, Dict[str, List[float]]]) -> Tuple[List[Dict], Dict[str, Dict[str, float]]]:
    summary_rows = []
    for expert_id, subset_scores in scores_by_expert.items():
        for subset in SUBSETS:
            stats = summarize_array(subset_scores[subset])
            row = {"expert_id": expert_id, "subset": subset}
            row.update(stats)
            summary_rows.append(row)

    variance = {}
    for subset in SUBSETS:
        variance[subset] = {}
        subset_rows = [row for row in summary_rows if row["subset"] == subset]
        for stat_name in STAT_NAMES:
            values = np.asarray([row[stat_name] for row in subset_rows], dtype=np.float64)
            values = values[~np.isnan(values)]
            variance[subset][stat_name] = float(np.var(values)) if values.size else float("nan")

    return summary_rows, variance


def write_csv(path: Path, rows: List[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_violin(scores_by_expert: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"all": "#4c78a8", "in_task": "#54a24b", "out_task": "#e45756"}
    data, positions, facecolors = [], [], []
    expert_ids = sorted(scores_by_expert)
    for expert_pos, expert_id in enumerate(expert_ids):
        for offset, subset in enumerate(SUBSETS):
            values = scores_by_expert[expert_id][subset]
            if values:
                data.append(values)
                positions.append(expert_pos * 4 + offset)
                facecolors.append(colors[subset])

    if not data:
        return

    fig_width = max(8, len(expert_ids) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    violin = ax.violinplot(data, positions=positions, widths=0.8, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], facecolors):
        body.set_facecolor(color)
        body.set_edgecolor("#222222")
        body.set_alpha(0.7)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#222222")

    ax.set_title("Per-expert max response distributions")
    ax.set_xlabel("expert_id")
    ax.set_ylabel("max_score")
    ax.set_xticks([i * 4 + 1 for i in range(len(expert_ids))])
    ax.set_xticklabels([str(expert_id) for expert_id in expert_ids])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=[Patch(facecolor=colors[subset], label=subset) for subset in SUBSETS], frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_box(scores_by_expert: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"all": "#4c78a8", "in_task": "#54a24b", "out_task": "#e45756"}
    data, positions, facecolors = [], [], []
    expert_ids = sorted(scores_by_expert)
    for expert_pos, expert_id in enumerate(expert_ids):
        for offset, subset in enumerate(SUBSETS):
            values = scores_by_expert[expert_id][subset]
            if values:
                data.append(values)
                positions.append(expert_pos * 4 + offset)
                facecolors.append(colors[subset])

    if not data:
        return

    fig_width = max(8, len(expert_ids) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    box = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], facecolors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#222222")
    for median in box["medians"]:
        median.set_color("#222222")

    ax.set_title("Per-expert max response box plot")
    ax.set_xlabel("expert_id")
    ax.set_ylabel("max_score")
    ax.set_xticks([i * 4 + 1 for i in range(len(expert_ids))])
    ax.set_xticklabels([str(expert_id) for expert_id in expert_ids])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=[Patch(facecolor=colors[subset], label=subset) for subset in SUBSETS], frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_histograms(scores_by_expert: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    import math
    import matplotlib.pyplot as plt

    colors = {"all": "#4c78a8", "in_task": "#54a24b", "out_task": "#e45756"}
    expert_ids = sorted(scores_by_expert)
    ncols = min(4, max(1, len(expert_ids)))
    nrows = int(math.ceil(len(expert_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for axis_idx, expert_id in enumerate(expert_ids):
        ax = axes[axis_idx // ncols][axis_idx % ncols]
        for subset in SUBSETS:
            values = scores_by_expert[expert_id][subset]
            if values:
                ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.4, color=colors[subset], label=subset)
        ax.set_title(f"expert {expert_id}")
        ax.set_xlabel("max_score")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2)
        if axis_idx == 0:
            ax.legend(frameon=False)

    for axis_idx in range(len(expert_ids), nrows * ncols):
        axes[axis_idx // ncols][axis_idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ecdf(scores_by_expert: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    import math
    import matplotlib.pyplot as plt

    colors = {"all": "#4c78a8", "in_task": "#54a24b", "out_task": "#e45756"}
    expert_ids = sorted(scores_by_expert)
    ncols = min(4, max(1, len(expert_ids)))
    nrows = int(math.ceil(len(expert_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for axis_idx, expert_id in enumerate(expert_ids):
        ax = axes[axis_idx // ncols][axis_idx % ncols]
        for subset in SUBSETS:
            values = np.asarray(scores_by_expert[expert_id][subset], dtype=np.float64)
            if values.size:
                values = np.sort(values)
                y = np.arange(1, values.size + 1) / values.size
                ax.plot(values, y, linewidth=1.4, color=colors[subset], label=subset)
        ax.set_title(f"expert {expert_id}")
        ax.set_xlabel("max_score")
        ax.set_ylabel("ECDF")
        ax.grid(alpha=0.2)
        if axis_idx == 0:
            ax.legend(frameon=False)

    for axis_idx in range(len(expert_ids), nrows * ncols):
        axes[axis_idx // ncols][axis_idx % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_plots(scores_by_expert: Dict[int, Dict[str, List[float]]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        plot_violin(scores_by_expert, output_dir / "expert_response_violin.png")
        plot_box(scores_by_expert, output_dir / "expert_response_box.png")
        plot_histograms(scores_by_expert, output_dir / "expert_response_hist.png")
        plot_ecdf(scores_by_expert, output_dir / "expert_response_ecdf.png")
    except ImportError as exc:
        logging.warning("Skipping plots because matplotlib is not available: %s", exc)


def default_output_dir(checkpoint_dir: Optional[Path], checkpoint_path: Path, task_id: int) -> Path:
    if checkpoint_dir is not None:
        return checkpoint_dir / "expert_response_analysis" / f"task_{task_id}"
    return checkpoint_path.parent / f"{checkpoint_path.stem}_expert_response_analysis"


def run_one_seed(config: Dict, cli_args: argparse.Namespace, seed: int, multiple_seeds: bool) -> Dict:
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
    rows, scores_by_expert = collect_expert_responses(
        model=model,
        loader=loader,
        class_to_task=class_to_task,
        task_offsets=task_offsets,
        apply_scale=not cli_args.no_scale,
    )
    summary_rows, variance = build_summary(scores_by_expert)

    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else default_output_dir(checkpoint_dir, checkpoint_path, model._cur_task)
    if multiple_seeds:
        output_dir = output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "expert_response_records.csv",
        rows,
        fieldnames=[
            "sample_offset",
            "sample_index",
            "true_class",
            "true_task",
            "expert_id",
            "is_in_task",
            "max_score",
            "argmax_class",
            "top1_score",
            "top2_score",
            "top2_class",
        ],
    )
    write_csv(
        output_dir / "expert_response_summary.csv",
        summary_rows,
        fieldnames=["expert_id", "subset", "count", *STAT_NAMES],
    )
    with (output_dir / "expert_stat_variance.json").open("w") as f:
        json.dump(variance, f, indent=2)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "seed": seed,
                "model_name": args["model_name"],
                "dataset": args["dataset"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_cnn_accy": checkpoint.get("cnn_accy"),
                "task_id": model._cur_task,
                "known_classes": model._known_classes,
                "total_classes": model._total_classes,
                "num_experts": model._cur_task + 1,
                "applied_scale": not cli_args.no_scale,
                "subsets": list(SUBSETS),
            },
            f,
            indent=2,
        )

    if not cli_args.skip_plots:
        write_plots(scores_by_expert, output_dir)

    logging.info("Saved expert response analysis to %s", output_dir)
    return {
        "seed": seed,
        "task_id": model._cur_task,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "num_records": len(rows),
        "summary": summary_rows,
        "expert_stat_variance": variance,
    }


def main() -> None:
    cli_args = setup_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [expert_response_analysis] %(message)s")
    config = load_json(cli_args.config)

    seeds = iter_config_seeds(config, cli_args.seed)
    if cli_args.checkpoint and len(seeds) > 1 and cli_args.seed is None:
        logging.warning("--checkpoint was set with multiple config seeds; analyzing the first seed only.")
        seeds = seeds[:1]

    results = []
    for seed in seeds:
        results.append(run_one_seed(config, cli_args, seed=seed, multiple_seeds=len(seeds) > 1))

    if len(results) > 1 and cli_args.output_dir:
        output_dir = Path(cli_args.output_dir)
        with (output_dir / "all_seed_outputs.json").open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
