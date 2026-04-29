#!/usr/bin/env python3
"""
Final-task resource footprint for incremental-learning tables.

This script reports the footprint after learning the final task:
  Params: all persistent nn.Parameter scalars, including the frozen ViT backbone.
  State: persistent non-parameter scalars required after the final task.
  Recomp.: non-parameter buffers that can be recomputed from existing parameters.

Values are scalar counts, not bytes, so they are independent of numeric precision.
The formulas intentionally follow the model code paths used by the experiment JSONs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


D = 768
H = 3072
L = 12
VIT_B_PARAMS = 85_795_584  # ViT-B/16 backbone without classifier head.


DATASET_CLASSES = {
    "cifar224": 100,
    "cub": 200,
    "domainnet": 200,
    "imagenetr": 200,
    "omnibenchmark": 300,
}


@dataclass
class Footprint:
    method: str
    dataset: str
    classes: int
    tasks: int
    params: int
    state: int = 0
    recomputable: int = 0
    flops: float | None = None
    notes: list[str] = field(default_factory=list)
    components: dict[str, int] = field(default_factory=dict)


def fmt_m(n: int) -> str:
    return f"{n / 1_000_000:.2f}"


def task_sizes(total_classes: int, init_cls: int, increment: int) -> list[int]:
    sizes = [int(init_cls)]
    while sum(sizes) + int(increment) < int(total_classes):
        sizes.append(int(increment))
    tail = int(total_classes) - sum(sizes)
    if tail > 0:
        sizes.append(tail)
    return sizes


def total_classes(args: dict) -> int:
    if "nb_classes" in args:
        return int(args["nb_classes"])
    dataset = str(args["dataset"]).lower()
    if dataset not in DATASET_CLASSES:
        raise ValueError(f"Unknown class count for dataset={dataset!r}; pass nb_classes in the config.")
    return DATASET_CLASSES[dataset]


def config_shape(args: dict) -> tuple[int, list[int]]:
    c = total_classes(args)
    return c, task_sizes(c, int(args["init_cls"]), int(args["increment"]))


def linear(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return in_dim * out_dim + (out_dim if bias else 0)


def cosine_linear(in_dim: int, out_dim: int, sigma: bool = True) -> int:
    return in_dim * out_dim + (1 if sigma else 0)


def continual_linear(task_sizes_: list[int], in_dim: int = D, bias: bool = True) -> int:
    return sum(linear(in_dim, size, bias=bias) for size in task_sizes_)


def adapter_params(ffn: int) -> int:
    return L * (2 * D * int(ffn) + int(ffn) + D)


def ssf_params() -> int:
    per_block = 2 * (3 * D + D + H + D + D + D)
    return L * per_block + 2 * D


def covariance_state(c: int, mode: str, rank: int = 0) -> int:
    mode = str(mode).lower()
    if mode == "variance":
        return c * D
    if mode == "diag_lowrank":
        return c * (D + D * int(rank) + int(rank))
    if mode == "covariance":
        return c * D * D
    raise ValueError(f"Unsupported covariance storage mode: {mode}")


def class_stats_state(c: int, mode: str, rank: int = 0) -> int:
    return c * D + covariance_state(c, mode, rank)


def default_cov_mode(args: dict, default: str = "covariance") -> str:
    return str(args.get("ca_storage_efficient_method", default)).lower()


def base_plus_head(c: int, bias: bool = True) -> int:
    return VIT_B_PARAMS + linear(D, c, bias=bias)


def count_l2p(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    pool = int(args.get("size", 10))
    length = int(args.get("length", 5))
    prompt = pool * length * D
    key = pool * D if bool(args.get("prompt_key", False)) else 0
    head = linear(D, c, bias=True)
    params = VIT_B_PARAMS + head + prompt + key
    return Footprint(
        method="L2P",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=params,
        components={
            "vit_backbone": VIT_B_PARAMS,
            "classifier_head": head,
            "prompt_pool": prompt,
            "prompt_keys": key,
        },
    )


def count_dualprompt(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    pool = int(args.get("size", 10))
    e_len = int(args.get("length", 5))
    g_len = int(args.get("g_prompt_length", 5))
    g_layers = len(args.get("g_prompt_layer_idx", [0, 1])) if args.get("use_g_prompt", True) else 0
    e_layers = len(args.get("e_prompt_layer_idx", [2, 3, 4])) if args.get("use_e_prompt", True) else 0
    g_factor = 2 if bool(args.get("use_prefix_tune_for_g_prompt", False)) else 1
    e_factor = 2 if bool(args.get("use_prefix_tune_for_e_prompt", False)) else 1
    g_prompt = g_layers * g_factor * g_len * D
    e_prompt = e_layers * e_factor * pool * e_len * D
    key = pool * D if bool(args.get("prompt_key", False)) else 0
    head = linear(D, c, bias=True)
    params = VIT_B_PARAMS + head + g_prompt + e_prompt + key
    return Footprint(
        method="DualPrompt",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=params,
        components={
            "vit_backbone": VIT_B_PARAMS,
            "classifier_head": head,
            "g_prompt": g_prompt,
            "e_prompt_pool": e_prompt,
            "prompt_keys": key,
        },
    )


def count_coda_prompt(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    pool, prompt_len, _ = args.get("prompt_param", [100, 8, 0])
    per_layer = int(pool) * int(prompt_len) * D + 2 * int(pool) * D
    prompt = 5 * per_layer
    head = linear(D, c, bias=True)
    return Footprint(
        method="CODA-Prompt",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + prompt + head,
        components={"vit_backbone": VIT_B_PARAMS, "coda_prompt": prompt, "classifier_head": head},
    )


def count_acil(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    b = int(args.get("buffer_size", 2048))
    state = D * b + b * c + b * b
    return Footprint(
        method="ACIL",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "random_buffer": D * b, "weight_buffer": b * c, "R": b * b},
    )


def count_aper(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    backbone_type = str(args.get("backbone_type", "")).lower()
    if "_ssf" in backbone_type:
        pet = ssf_params()
        variant = "ssf"
    elif "_vpt" in backbone_type:
        pet = L * int(args.get("prompt_token_num", 10)) * D
        variant = "vpt"
    elif "_adapter" in backbone_type or "adapter" in backbone_type:
        pet = adapter_params(int(args.get("ffn_num", 64)))
        variant = "adapter"
    else:
        pet = 0
        variant = "finetune"
    head = cosine_linear(D, c, sigma=True)
    return Footprint(
        method="APER",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + pet + head,
        components={"vit_backbone": VIT_B_PARAMS, f"{variant}_params": pet, "classifier_head": head},
    )


def count_slca(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    heads = continual_linear(sizes, D, bias=True)
    state = class_stats_state(c, "covariance")
    return Footprint(
        method="SLCA",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + heads,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "continual_heads": heads, "class_mean_cov": state},
    )


def count_ssiat(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    pet = adapter_params(int(args.get("ffn_num", 64)))
    heads = continual_linear(sizes, D, bias=False)
    state = class_stats_state(c, "covariance") + 1
    return Footprint(
        method="SSIAT",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + pet + heads,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "adapter": pet, "continual_heads": heads, "class_mean_cov_radius": state},
    )


def count_fecam(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    pet = adapter_params(int(args.get("ffn_num", 64)))
    head = cosine_linear(D, c, sigma=True)
    state = c * D * D
    return Footprint(
        method="FeCAM",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + pet + head,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "adapter": pet, "classifier_head": head, "cov_mats": state},
    )


def count_ranpac(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    m = int(args.get("M", 10000))
    pet = adapter_params(int(args.get("ffn_num", 64))) if "adapter" in str(args.get("backbone_type", "")).lower() else 0
    head = cosine_linear(m, c, sigma=True)
    state = D * m + m * c + m * m
    return Footprint(
        method="RanPAC",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + pet + head,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "adapter": pet, "rp_classifier": head, "W_rand_Q_G": state},
    )


def count_min(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    b = int(args.get("buffer_size", 16384))
    noise = L * 4 * D
    normal_fc = linear(b, c, bias=False)
    state = D * b + b * c + b * b + len(sizes) * D
    return Footprint(
        method="MiN",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + noise + normal_fc,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "noise_params": noise, "normal_fc": normal_fc, "buffer_weight_R_prototypes": state},
    )


def count_ease(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    ffn = int(args.get("ffn_num", 16))
    per_adapter = adapter_params(ffn)
    adapter_total = len(sizes) * per_adapter
    sources = len(sizes) + (1 if bool(args.get("use_init_ptm", True)) else 0)
    head = cosine_linear(sources * D, c, sigma=True)
    proxy = cosine_linear(D, sizes[-1], sigma=True)
    return Footprint(
        method="EASE",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + adapter_total + head + proxy,
        components={"vit_backbone": VIT_B_PARAMS, "adapters": adapter_total, "classifier_head": head, "proxy_fc": proxy},
    )


def count_mos(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    ffn = int(args.get("ffn_num", 16))
    per_adapter = adapter_params(ffn)
    adapter_total = (len(sizes) + 1) * per_adapter
    if float(args.get("adapter_momentum", 0.0) or 0.0) > 0:
        adapter_total += per_adapter
    head = cosine_linear(D, c, sigma=True)
    mode = default_cov_mode(args, "variance")
    state = class_stats_state(c, mode) + c  # cls2task mapping.
    return Footprint(
        method="MOS",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + adapter_total + head,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "adapters_and_ema": adapter_total, "classifier_head": head, "class_stats_and_router": state},
    )


def count_tuna(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    ffn = int(args.get("ffn_num", 16))
    per_adapter = adapter_params(ffn)
    adapter_total = (len(sizes) + 2) * per_adapter  # historical adapters + current + merged.
    head = continual_linear(sizes, D, bias=False)
    shared = continual_linear(sizes, D, bias=False) if bool(args.get("enable_shared_cls_classifier", False)) else 0
    calib = 2 * len(sizes) if bool(args.get("enable_expert_calibration", False)) else 0
    mode = default_cov_mode(args, "variance")
    state = class_stats_state(c, mode)
    return Footprint(
        method="TUNA",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + adapter_total + head + shared + calib,
        state=state,
        components={"vit_backbone": VIT_B_PARAMS, "adapters_current_merged": adapter_total, "classifier_heads": head + shared + calib, "class_stats": state},
    )


def snapshot_sum(sizes: list[int], include_task0: bool = True) -> int:
    total = 0
    seen = 0
    for idx, size in enumerate(sizes):
        seen += size
        if idx == 0 and not include_task0:
            continue
        total += VIT_B_PARAMS + linear(D, seen, bias=True)
    return total


def snapshot_sum_after_task(sizes: list[int], start_task: int = 0) -> int:
    total = 0
    seen = 0
    for idx, size in enumerate(sizes):
        seen += size
        if idx < start_task:
            continue
        total += VIT_B_PARAMS + linear(D, seen, bias=True)
    return total


def count_cofima(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    heads = continual_linear(sizes, D, bias=True)
    stats = class_stats_state(c, "covariance")
    prev_nets = snapshot_sum(sizes, include_task0=True)
    init_nets = snapshot_sum_after_task(sizes, start_task=1)
    fisher = snapshot_sum(sizes, include_task0=True) if bool(args.get("fisher_weighting", False)) else 0
    state = stats + prev_nets + init_nets + fisher
    return Footprint(
        method="COFiMA",
        dataset=args["dataset"],
        classes=c,
        tasks=len(sizes),
        params=VIT_B_PARAMS + heads,
        state=state,
        components={
            "vit_backbone": VIT_B_PARAMS,
            "continual_heads": heads,
            "class_mean_cov": stats,
            "prev_nets": prev_nets,
            "init_nets": init_nets,
            "fisher": fisher,
        },
        notes=["State counts all checkpoint dictionaries kept by the current implementation."],
    )


def count_spie(args: dict) -> Footprint:
    c, sizes = config_shape(args)
    t = len(sizes)
    vera_rank = int(args.get("vera_rank", 256))
    expert_tokens = int(args.get("expert_tokens", 4))
    per_expert_adapter = L * ((vera_rank + H) + (vera_rank + D))
    expert_adapters = (t + 1) * per_expert_adapter
    token_params = (t + 1) * expert_tokens * D
    fc_shared = continual_linear(sizes, D, bias=False)
    expert_heads = continual_linear(sizes, 2 * D, bias=False)
    state = class_stats_state(c, "diag_lowrank", 4)
    backbone = VIT_B_PARAMS
    return Footprint(
        method="SPIE",
        dataset=args["dataset"],
        classes=c,
        tasks=t,
        params=backbone + expert_adapters + token_params + fc_shared + expert_heads,
        state=state,
        components={
            "vit_backbone": backbone,
            "expert_vera_adapters": expert_adapters,
            "expert_tokens": token_params,
            "fc_shared_cls": fc_shared,
            "expert_heads": expert_heads,
            "shared_class_stats": state,
        },
    )


COUNTERS: dict[str, Callable[[dict], Footprint]] = {
    "acil": count_acil,
    "aper": count_aper,
    "aper_adapter": count_aper,
    "aper_ssf": count_aper,
    "aper_vpt": count_aper,
    "coda_prompt": count_coda_prompt,
    "cofima": count_cofima,
    "dualprompt": count_dualprompt,
    "ease": count_ease,
    "fecam": count_fecam,
    "l2p": count_l2p,
    "min": count_min,
    "mos": count_mos,
    "ranpac": count_ranpac,
    "slca": count_slca,
    "spie": count_spie,
    "ssiat": count_ssiat,
    "tuna": count_tuna,
}


FLOPS_BY_METHOD = {
    # Final-task inference totals measured by tools/count_flops.py for DomainNet-like 10-task configs.
    # Use --no-flops if you do not want these reference values.
    "acil": 16.87,
    "aper": 17.81,
    "aper_adapter": 17.81,
    "coda_prompt": 35.17,
    "cofima": 16.87,
    "dualprompt": 35.17,
    "ease": 17.81,
    "fecam": 17.81,
    "l2p": 37.48,
    "min": 18.27,
    "mos": 193.40,
    "ranpac": 19.42,
    "slca": 16.87,
    "spie": 35.45,
    "ssiat": 17.81,
    "tuna": 193.40,
}


def count_config(path: Path, include_flops: bool = False) -> Footprint:
    args = json.loads(path.read_text())
    name = str(args["model_name"]).lower()
    if name not in COUNTERS:
        raise NotImplementedError(f"No footprint counter for model_name={name!r} ({path})")
    fp = COUNTERS[name](args)
    if include_flops:
        fp.flops = FLOPS_BY_METHOD.get(name)
    fp.notes = [note for note in fp.notes if note]
    return fp


def print_markdown(rows: list[Footprint]) -> None:
    print("| Method | Dataset | C | T | Params (M) | State (M) | FLOPs (G) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        flops = "" if row.flops is None else f"{row.flops:.2f}"
        print(
            f"| {row.method} | {row.dataset} | {row.classes} | {row.tasks} | "
            f"{fmt_m(row.params)} | {fmt_m(row.state)} | {flops} |"
        )


def write_csv(rows: list[Footprint], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "dataset",
                "classes",
                "tasks",
                "params",
                "state",
                "recomputable",
                "params_m",
                "state_m",
                "recomputable_m",
                "flops_g",
                "notes",
                "components",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "dataset": row.dataset,
                    "classes": row.classes,
                    "tasks": row.tasks,
                    "params": row.params,
                    "state": row.state,
                    "recomputable": row.recomputable,
                    "params_m": fmt_m(row.params),
                    "state_m": fmt_m(row.state),
                    "recomputable_m": fmt_m(row.recomputable),
                    "flops_g": "" if row.flops is None else f"{row.flops:.2f}",
                    "notes": " ".join(row.notes),
                    "components": json.dumps(row.components, sort_keys=True),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="Experiment JSON files.")
    parser.add_argument("--dataset", default="domainnet", help="Use exps/<dataset>/*.json when configs are omitted.")
    parser.add_argument("--csv", type=Path, help="Optional CSV output path.")
    parser.add_argument(
        "--with-reference-flops",
        action="store_true",
        help="Attach reference FLOPs values measured by tools/count_flops.py for DomainNet-like 10-task configs.",
    )
    args = parser.parse_args()

    configs = args.configs
    if not configs:
        configs = sorted(Path("exps").joinpath(args.dataset).glob("*.json"))
    rows = [count_config(path, include_flops=args.with_reference_flops) for path in configs]
    rows.sort(key=lambda row: row.method.lower())
    print_markdown(rows)
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
