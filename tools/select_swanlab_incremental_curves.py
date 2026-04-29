#!/usr/bin/env python3
"""Select the paper-table matching SwanLab curves for plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_LABELS = {
    "imagenetr": "ImageNet-R",
    "omnibenchmark": "Omnibenchmark",
    "domainnet": "DomainNet",
}

METHOD_ORDER = [
    # "L2P",
    # "DualPrompt",
    "CODA-Prompt",
    # "ACIL",
    # "SLCA",
    # "SSIAT",
    "FeCAM",
    # "RanPAC",
    "APER",
    "EASE",
    "COFiMA",
    "MOS",
    "TUNA",
    "MIN",
    "SPIE",
]

# Values in tables/main_results.tex. They are used only to disambiguate duplicate
# SwanLab runs; the plotted y-values are still the per-step eval/cnn/top1 curves.
TARGETS = {
    ("imagenetr", "L2P"): (77.97, 72.32),
    ("imagenetr", "DualPrompt"): (75.01, 69.15),
    ("imagenetr", "ACIL"): (73.25, 68.95),
    ("imagenetr", "SLCA"): (85.28, 80.88),
    ("imagenetr", "SSIAT"): (83.66, 79.95),
    ("imagenetr", "FeCAM"): (76.02, 69.40),
    ("imagenetr", "RanPAC"): (83.42, 78.28),
    ("imagenetr", "APER"): (78.90, 72.80),
    ("imagenetr", "MOS"): (81.57, 75.47),
    ("imagenetr", "TUNA"): (84.13, 79.32),
    ("imagenetr", "MIN"): (84.87, 79.20),
    ("imagenetr", "SPIE"): (84.42, 78.93),
    ("omnibenchmark", "L2P"): (73.28, 63.89),
    ("omnibenchmark", "DualPrompt"): (73.10, 63.01),
    ("omnibenchmark", "CODA-Prompt"): (76.39, 67.69),
    ("omnibenchmark", "ACIL"): (82.23, 74.54),
    ("omnibenchmark", "SLCA"): (82.23, 74.82),
    ("omnibenchmark", "SSIAT"): (84.39, 77.61),
    ("omnibenchmark", "FeCAM"): (83.54, 76.94),
    ("omnibenchmark", "RanPAC"): (84.24, 78.20),
    ("omnibenchmark", "APER"): (80.77, 74.42),
    ("omnibenchmark", "EASE"): (80.63, 74.04),
    ("omnibenchmark", "COFiMA"): (83.20, 76.07),
    ("omnibenchmark", "MOS"): (82.30, 75.56),
    ("omnibenchmark", "TUNA"): (82.65, 74.54),
    ("omnibenchmark", "MIN"): (86.39, 80.27),
    ("omnibenchmark", "SPIE"): (82.06, 76.11),
    ("domainnet", "L2P"): (80.51, 75.00),
    ("domainnet", "DualPrompt"): (83.75, 77.63),
    ("domainnet", "CODA-Prompt"): (84.09, 78.13),
    ("domainnet", "ACIL"): (88.65, 83.68),
    ("domainnet", "SLCA"): (89.47, 85.15),
    ("domainnet", "SSIAT"): (89.81, 84.18),
    ("domainnet", "FeCAM"): (84.28, 79.15),
    ("domainnet", "RanPAC"): (90.80, 87.18),
    ("domainnet", "APER"): (84.36, 78.98),
    ("domainnet", "EASE"): (84.71, 78.92),
    ("domainnet", "COFiMA"): (90.03, 86.02),
    ("domainnet", "MOS"): (87.29, 81.67),
    ("domainnet", "TUNA"): (88.91, 82.79),
    ("domainnet", "MIN"): (91.35, 87.60),
    ("domainnet", "SPIE"): (90.19, 86.38),
}


def variant_label(row: dict) -> str:
    method = row["method"]
    if method != "SPIE":
        return method
    name = row["name"].lower()
    config = str(row.get("config", "")).lower()
    if "lowrank4" in name or "lowrank4" in config:
        return "SPIE"
    if "lowrank" in name or "lowrank" in config:
        return "SPIE-other-rank"
    return "SPIE"


def distance(row: dict, target: tuple[float, float]) -> float:
    avg, final = target
    return abs(row["computed_avg_top1"] - avg) + abs(row["computed_final_top1"] - final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("figures/swanlab_incremental_curves_raw.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/swanlab_incremental_curves_selected.json"))
    args = parser.parse_args()

    rows = json.loads(args.raw.read_text(encoding="utf-8"))
    selected: dict[str, dict[str, list[float]]] = {label: {} for label in DATASET_LABELS.values()}
    selected_meta = []

    for dataset, dataset_label in DATASET_LABELS.items():
        for method in METHOD_ORDER:
            target = TARGETS.get((dataset, method))
            candidates = [
                row
                for row in rows
                if row["dataset"] == dataset
                and variant_label(row) == method
                and len(row["curve"]) == int(row["nb_tasks"] or 10)
            ]
            if not candidates:
                continue
            if target is None:
                best = min(
                    candidates,
                    key=lambda row: (
                        not str(row.get("name", "")).startswith("official-"),
                        str(row.get("created_at", "")),
                    ),
                )
                target = (best["computed_avg_top1"], best["computed_final_top1"])
            else:
                best = min(candidates, key=lambda row: distance(row, target))
            selected[dataset_label][method] = best["curve"]
            selected_meta.append(
                {
                    "dataset": dataset_label,
                    "method": method,
                    "name": best["name"],
                    "path": best["path"],
                    "computed_avg_top1": best["computed_avg_top1"],
                    "computed_final_top1": best["computed_final_top1"],
                    "target_avg_top1": target[0],
                    "target_final_top1": target[1],
                }
            )

    payload = {
        "method_order": METHOD_ORDER,
        "curves": selected,
        "selected_runs": selected_meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"selected={sum(len(v) for v in selected.values())} output={args.output}")


if __name__ == "__main__":
    main()
