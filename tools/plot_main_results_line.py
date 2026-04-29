#!/usr/bin/env python3
"""Plot incremental accuracy curves for the SPIE main experiments.

Each subplot is one dataset. The x-axis is the incremental step expressed as
the number of seen classes, and the y-axis is top-1 accuracy at that step.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

cache_root = Path(tempfile.gettempdir()) / "spie-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib.pyplot as plt
import numpy as np


DATASETS = {
    "ImageNet-R": {"init_cls": 20, "increment": 20, "nb_tasks": 10},
    "Omnibenchmark": {"init_cls": 30, "increment": 30, "nb_tasks": 10},
    "DomainNet": {"init_cls": 20, "increment": 20, "nb_tasks": 10},
}

METHODS = [
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

STYLES = {
    "L2P": ("#8E63CE", "o"),
    "DualPrompt": ("#7F7F7F", "v"),
    "CODA-Prompt": ("#B07AA1", "^"),
    "ACIL": ("#59A14F", "P"),
    "SLCA": ("#9467BD", "D"),
    "SSIAT": ("#FF9DA7", "X"),
    "FeCAM": ("#B6992D", "o"),
    "RanPAC": ("#4ECDC4", "o"),
    "APER": ("#8C564B", "o"),
    "EASE": ("#8A9A22", "o"),
    "COFiMA": ("#FF7F0E", "o"),
    "MOS": ("#4C78A8", "o"),
    "TUNA": ("#2CA02C", "o"),
    "MIN": ("#E15759", "s"),
    "SPIE": ("#D62728", "s"),
}

CURVES = {
    "ImageNet-R": {
        "CODA-Prompt": [90.57, 87.96, 84.74, 82.19, 80.27, 78.79, 77.98, 78.07, 77.33, 75.18],
        "FeCAM": [92.31, 83.5, 79.45, 76.82, 73.84, 72.62, 71.48, 70.87, 69.93, 69.4],
        "APER": [91.73, 84.71, 82.2, 80.26, 77.62, 76.21, 75.44, 74.57, 73.47, 72.8],
        "EASE": [93.32, 87.96, 85.94, 83.71, 81.57, 80.69, 79.85, 79.08, 78.2, 77.4],
        "COFiMA": [94.78, 88.95, 87.55, 86.39, 83.95, 82.66, 82.01, 81.85, 81.18, 80.32],
        "MOS": [92.02, 88.72, 84.9, 82.63, 80.4, 79.12, 78.33, 77.71, 76.44, 75.47],
        "TUNA": [92.74, 88.8, 87.18, 85.07, 82.78, 82.06, 81.82, 81.43, 80.06, 79.32],
        "MIN": [95.79, 89.54, 87.12, 85.31, 82.58, 81.41, 80.59, 80.12, 78.08, 78.2],
        "SPIE": [95.94, 91.16, 88.65, 86.31, 84.02, 83.0, 81.9, 81.58, 81.75, 79.93],
    },
    "Omnibenchmark": {
        "CODA-Prompt": [93.0, 85.74, 81.54, 77.45, 75.75, 73.08, 72.17, 69.47, 68.0, 67.69],
        "FeCAM": [92.67, 91.49, 89.38, 85.59, 83.3, 80.9, 79.85, 78.34, 76.98, 76.94],
        "APER": [89.5, 89.32, 86.43, 82.71, 80.49, 77.81, 77.08, 75.4, 74.55, 74.42],
        "EASE": [90.17, 88.82, 86.15, 82.67, 80.36, 77.87, 76.92, 75.08, 74.23, 74.04],
        "COFiMA": [94.0, 90.58, 88.1, 85.01, 83.37, 80.9, 79.54, 77.44, 77.02, 76.07],
        "MOS": [91.17, 91.49, 88.1, 84.09, 81.9, 79.62, 78.56, 76.8, 75.68, 75.56],
        "TUNA": [93.0, 89.24, 88.54, 84.8, 83.0, 80.85, 78.97, 77.38, 76.22, 74.54],
        "MIN": [95.33, 93.83, 92.1, 88.35, 86.04, 83.44, 82.6, 81.52, 80.42, 80.27],
        "SPIE": [93.17, 88.66, 86.93, 83.59, 81.53, 79.34, 78.18, 77.01, 76.09, 76.11],
    },
    "DomainNet": {
        "CODA-Prompt": [95.14, 91.15, 86.06, 85.32, 84.16, 81.29, 80.82, 79.8, 79.06, 78.13],
        "FeCAM": [93.19, 90.32, 84.95, 84.84, 84.0, 82.51, 81.88, 81.53, 80.43, 79.15],
        "APER": [93.79, 90.37, 85.8, 85.29, 84.21, 82.28, 81.42, 81.21, 80.24, 78.98],
        "EASE": [94.74, 92.01, 87.38, 86.02, 84.69, 81.57, 80.95, 80.74, 80.04, 78.92],
        "COFiMA": [96.69, 94.67, 90.48, 90.99, 90.13, 88.33, 87.89, 87.73, 87.41, 86.02],
        "MOS": [96.19, 93.36, 88.92, 88.45, 87.33, 85.29, 84.46, 84.02, 83.17, 81.67],
        "TUNA": [96.94, 94.69, 90.42, 90.4, 89.47, 87.28, 86.43, 85.9, 84.78, 82.79],
        "MIN": [97.34, 94.59, 91.25, 91.02, 90.22, 89.01, 88.42, 88.41, 86.65, 86.6],
        "SPIE": [96.69, 95.49, 91.78, 91.94, 91.11, 89.46, 89.35, 89.17, 88.5, 87.38],
    },
}


def seen_classes(dataset: str) -> np.ndarray:
    cfg = DATASETS[dataset]
    return np.array(
        [cfg["init_cls"] + step * cfg["increment"] for step in range(cfg["nb_tasks"])],
        dtype=int,
    )


def validate_curves(curves: dict[str, dict[str, list[float]]]) -> None:
    errors = []
    for dataset, method_curves in curves.items():
        expected = DATASETS[dataset]["nb_tasks"]
        for method, curve in method_curves.items():
            if len(curve) != expected:
                errors.append(f"{dataset}/{method}: expected {expected} points, got {len(curve)}")
    if errors:
        raise ValueError("Invalid curve lengths:\n" + "\n".join(errors))


def plot_curves(
    output: Path,
    methods: list[str] | None = None,
) -> None:
    validate_curves(CURVES)
    methods = methods or METHODS

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15.5, 4.2), sharey=False)
    if len(DATASETS) == 1:
        axes = [axes]

    legend_handles = {}
    for ax, dataset in zip(axes, DATASETS):
        x = seen_classes(dataset)
        plotted = False
        for method in methods:
            curve = CURVES.get(dataset, {}).get(method)
            if not curve:
                continue

            color, marker = STYLES.get(method, ("#333333", "o"))
            is_spie = method.startswith("SPIE")
            linewidth = 2.6 if is_spie else 1.5
            markersize = 5.0 if is_spie else 3.3
            alpha = 1.0 if is_spie else 0.78
            (line,) = ax.plot(
                x,
                curve,
                label=method,
                color=color,
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                alpha=alpha,
            )
            legend_handles.setdefault(method, line)
            plotted = True

        ax.set_title(dataset)
        ax.set_xlabel("Number of classes")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        if not plotted:
            ax.set_ylim(60, 100)
            ax.text(
                0.5,
                0.5,
                "fill CURVES",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
                fontsize=11,
            )
        ax.grid(True, linestyle=(0, (4, 4)), linewidth=0.8, color="#AFAFAF", alpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend_handles:
        ordered = [method for method in methods if method in legend_handles]
        fig.legend(
            [legend_handles[method] for method in ordered],
            ordered,
            loc="lower center",
            ncol=min(len(ordered), 8),
            frameon=False,
        )
        bottom = 0.20
    else:
        bottom = 0.08
        fig.text(
            0.5,
            0.5,
            "No curves found",
            ha="center",
            va="center",
            fontsize=12,
        )

    fig.tight_layout(rect=(0, bottom, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/main_results_incremental_curves.pdf"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional subset/order of methods to draw, e.g. --methods FeCAM APER EASE COFiMA MOS MIN SPIE",
    )
    args = parser.parse_args()
    plot_curves(args.output, methods=args.methods)


if __name__ == "__main__":
    main()
