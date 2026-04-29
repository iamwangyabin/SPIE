#!/usr/bin/env python3
"""Fetch per-step top-1 curves from SwanLab for SPIE main-result plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from swanlab import Api


DATASETS = {"imagenetr", "omnibenchmark", "domainnet"}
METHODS = {
    "l2p": "L2P",
    "dualprompt": "DualPrompt",
    "coda_prompt": "CODA-Prompt",
    "acil": "ACIL",
    "slca": "SLCA",
    "ssiat": "SSIAT",
    "fecam": "FeCAM",
    "ranpac": "RanPAC",
    "aper": "APER",
    "ease": "EASE",
    "cofima": "COFiMA",
    "mos": "MOS",
    "tuna": "TUNA",
    "min": "MIN",
    "spie": "SPIE",
}


def is_spie_rank4(run: Any, cfg: dict) -> bool:
    text = " ".join(
        str(value).lower()
        for value in (
            getattr(run, "name", ""),
            cfg.get("config"),
            cfg.get("prefix"),
            cfg.get("note"),
        )
        if value is not None
    )
    return "lowrank4" in text or "rank4" in text


def metric_curve(run: Any) -> list[float]:
    try:
        df = run.metrics(keys=["eval/cnn/top1"], x_axis="step", sample=1000)
    except Exception:
        return []
    if df is None or "eval/cnn/top1" not in df:
        return []
    return [float(v) for v in df["eval/cnn/top1"].dropna().tolist()]


def summary_metrics(run: Any) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "final_avg_top1": None,
        "final_top1": None,
    }
    try:
        df = run.metrics(
            keys=["summary/cnn/final_avg_top1", "summary/cnn/final_top1"],
            x_axis="step",
            sample=1000,
        )
    except Exception:
        return result
    if df is None:
        return result
    mapping = {
        "summary/cnn/final_avg_top1": "final_avg_top1",
        "summary/cnn/final_top1": "final_top1",
    }
    for column, out_key in mapping.items():
        if column not in df:
            continue
        values = df[column].dropna().tolist()
        if values:
            result[out_key] = float(values[0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="iamwan/SPIE")
    parser.add_argument("--output", type=Path, default=Path("figures/swanlab_incremental_curves_raw.json"))
    parser.add_argument("--only-finished", action="store_true", default=True)
    args = parser.parse_args()

    if not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("SWANLAB_API_KEY is required in the environment.")

    api = Api()
    runs = list(api.runs(path=args.project))
    records = []
    for run in runs:
        cfg = getattr(run.profile, "config", {}) or {}
        dataset = cfg.get("dataset")
        model_name = cfg.get("model_name")
        if dataset not in DATASETS or model_name not in METHODS:
            continue
        if model_name == "spie" and not is_spie_rank4(run, cfg):
            continue
        if args.only_finished and getattr(run, "state", "") != "FINISHED":
            continue

        curve = metric_curve(run)
        if not curve:
            continue
        summaries = summary_metrics(run)
        records.append(
            {
                "id": run.id,
                "name": run.name,
                "path": run.path,
                "state": run.state,
                "show": getattr(run, "show", None),
                "created_at": str(getattr(run, "created_at", "")),
                "finished_at": str(getattr(run, "finished_at", "")),
                "dataset": dataset,
                "method": METHODS[model_name],
                "model_name": model_name,
                "seed": cfg.get("seed"),
                "init_cls": cfg.get("init_cls"),
                "increment": cfg.get("increment"),
                "nb_tasks": cfg.get("nb_tasks"),
                "prefix": cfg.get("prefix"),
                "config": cfg.get("config"),
                "note": cfg.get("note"),
                "curve": curve,
                "computed_avg_top1": round(sum(curve) / len(curve), 2),
                "computed_final_top1": curve[-1],
                **summaries,
            }
        )

    records.sort(key=lambda r: (r["dataset"], r["method"], r["created_at"], r["name"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"runs={len(runs)} matched_curves={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
