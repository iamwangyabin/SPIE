#!/usr/bin/env python3
"""
Adaptive training sweep for SPiE on ImageNet-R.

This script launches real training runs by calling:
    python -u main.py --config <generated_config.json> --note <run_note>

It is intentionally biased toward epoch-sensitive exploration:
1. coarse global epoch scaling
2. coordinate refinement for shared / expert / classifier-align epochs
3. coordinate refinement for branch learning rates
4. refinement for shape-distillation knobs
5. optional margin sweep

Artifacts are written under --output-dir:
- generated_configs/
- launcher_logs/
- sweep_state.json
- sweep_results.csv
- best_config.json
- best_result.json
- rerun_best.sh
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parent
AVERAGE_ACCURACY_RE = re.compile(r"Average Accuracy \(([^)]+)\):\s*([-+]?\d+(?:\.\d+)?)")
RUN_DIR_RE = re.compile(r"log dir:\s*(\S+)")
TASK_CKPT_RE = re.compile(r"task_(\d+)\.pkl$")

TRACKED_KEYS = [
    "task0_shared_epochs",
    "shared_cls_epochs",
    "task0_expert_epochs",
    "incremental_expert_epochs",
    "shared_cls_crct_epochs",
    "task0_shared_lr",
    "shared_cls_lr",
    "task0_expert_lr",
    "incremental_expert_lr",
    "shared_cls_ca_lr",
    "expert_shape_distill_lambda",
    "expert_shape_distill_temperature",
    "expert_shape_reg_cap_ratio",
    "m",
    "scale",
    "batch_size",
    "freeze_shared_lora_after_task0",
]

OBJECTIVE_FALLBACKS = {
    "avg_p_final_top1": ["avg_p_final_top1", "avg_cnn_top1", "final_p_final_top1", "final_cnn_top1"],
    "avg_cnn_top1": ["avg_cnn_top1", "avg_p_final_top1", "final_cnn_top1", "final_p_final_top1"],
    "final_p_final_top1": ["final_p_final_top1", "avg_p_final_top1", "final_cnn_top1", "avg_cnn_top1"],
    "final_cnn_top1": ["final_cnn_top1", "avg_cnn_top1", "final_p_final_top1", "avg_p_final_top1"],
}

SEARCH_PRESETS = {
    "quick": {
        "global_epoch_scales": [0.85, 1.0, 1.25],
        "shared_epoch_scales": [0.85, 1.0, 1.2],
        "expert_epoch_scales": [0.9, 1.0, 1.2],
        "ca_epoch_scales": [0.75, 1.0, 1.25],
        "shared_lr_scales": [],
        "expert_lr_scales": [0.9, 1.0, 1.1],
        "ca_lr_scales": [],
        "lambda_values": [0.05, 0.1, 0.15],
        "temperature_values": [],
        "cap_ratio_values": [],
        "margin_values": [],
    },
    "medium": {
        "global_epoch_scales": [0.8, 1.0, 1.25, 1.5],
        "shared_epoch_scales": [0.85, 1.0, 1.2],
        "expert_epoch_scales": [0.9, 1.0, 1.2],
        "ca_epoch_scales": [0.75, 1.0, 1.25],
        "shared_lr_scales": [0.85, 1.0, 1.15],
        "expert_lr_scales": [0.85, 1.0, 1.15],
        "ca_lr_scales": [0.8, 1.0, 1.2],
        "lambda_values": [0.05, 0.1, 0.15, 0.2],
        "temperature_values": [1.0, 2.0, 3.0],
        "cap_ratio_values": [],
        "margin_values": [0.0, 0.1, 0.2, 0.3],
    },
    "full": {
        "global_epoch_scales": [0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
        "shared_epoch_scales": [0.8, 0.9, 1.0, 1.15, 1.3],
        "expert_epoch_scales": [0.85, 1.0, 1.15, 1.3, 1.45],
        "ca_epoch_scales": [0.67, 0.85, 1.0, 1.25, 1.5],
        "shared_lr_scales": [0.8, 0.9, 1.0, 1.1, 1.25],
        "expert_lr_scales": [0.8, 0.9, 1.0, 1.1, 1.25],
        "ca_lr_scales": [0.7, 0.85, 1.0, 1.2, 1.4],
        "lambda_values": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
        "temperature_values": [1.0, 1.5, 2.0, 3.0, 4.0],
        "cap_ratio_values": [0.1, 0.2, 0.25, 0.35, 0.5],
        "margin_values": [0.0, 0.1, 0.2, 0.3, 0.4],
    },
}


def build_parser(
    *,
    description: str,
    default_base_config: str,
    default_output_dir: str,
    default_prefix: str,
    default_target_dataset: Optional[str],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-config", type=str, default=default_base_config, help="Base JSON config.")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Sweep artifact directory.")
    parser.add_argument("--preset", type=str, default="medium", choices=sorted(SEARCH_PRESETS.keys()))
    parser.add_argument(
        "--objective",
        type=str,
        default="avg_p_final_top1",
        choices=sorted(OBJECTIVE_FALLBACKS.keys()),
        help="Primary metric used to rank runs.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess runs.")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for subprocess training runs.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on newly launched training runs in this sweep invocation.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview candidates without launching training.")
    parser.add_argument(
        "--enable-swanlab",
        action="store_true",
        help="Keep SwanLab enabled from the base config. Default is to disable it for sweeps.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next candidate when a run fails.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Allow rerunning candidates that previously failed in the same output-dir state.",
    )
    parser.add_argument(
        "--force-prefix",
        type=str,
        default=default_prefix,
        help="Prefix used for generated training configs.",
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        default=default_target_dataset,
        help="Override dataset in generated configs. Defaults to the dataset already stored in the base config.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser(
        description="Adaptive SPiE ImageNet-R training sweep",
        default_base_config="exps/spie_inr.json",
        default_output_dir="sweep_spie_inr_train",
        default_prefix="spie-autosweep-inr",
        default_target_dataset="imagenetr",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def sanitize_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "default"
    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("-")
    return "".join(chars).strip("-") or "default"


def metric_name_to_key(name: str) -> str:
    normalized = sanitize_text(str(name).strip().lower())
    return normalized.replace("-", "_")


def float_str(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def stable_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def scaled_int(value: Any, factor: float, min_value: int = 1) -> int:
    return max(min_value, int(round(float(value) * float(factor))))


def scaled_float(value: Any, factor: float, digits: int = 8) -> float:
    return stable_float(float(value) * float(factor), digits=digits)


def config_fingerprint(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def init_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "configs": output_dir / "generated_configs",
        "launcher_logs": output_dir / "launcher_logs",
        "state": output_dir / "sweep_state.json",
        "results_csv": output_dir / "sweep_results.csv",
        "best_config": output_dir / "best_config.json",
        "best_result": output_dir / "best_result.json",
        "rerun_best": output_dir / "rerun_best.sh",
    }
    for key in ("root", "configs", "launcher_logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "runs": [],
        }
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def persist_state(paths: Dict[str, Path], state: Dict[str, Any], objective_key: str) -> None:
    write_json(paths["state"], state)
    write_results_csv(paths["results_csv"], state.get("runs", []))
    best = best_completed_result(state.get("runs", []), objective_key)
    if best is not None:
        write_json(paths["best_config"], best["config"])
        write_json(paths["best_result"], best)
        write_rerun_script(paths["rerun_best"], paths["best_config"], best.get("note", "best"))


def write_rerun_script(path: Path, config_path: Path, note: str) -> None:
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f'python -u main.py --config "{config_path}" --note "{sanitize_text(note)}-rerun"\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    os.chmod(path, 0o755)


def write_results_csv(path: Path, runs: Sequence[Dict[str, Any]]) -> None:
    metric_columns = [
        "avg_cnn_top1",
        "avg_nme_top1",
        "avg_shared_fc_top1",
        "avg_p_moe_top1",
        "avg_p_final_top1",
        "final_cnn_top1",
        "final_nme_top1",
        "final_shared_fc_top1",
        "final_p_moe_top1",
        "final_p_final_top1",
    ]
    fieldnames = [
        "run_id",
        "stage",
        "label",
        "status",
        "objective_value",
        "objective_used",
        "runtime_sec",
        "exit_code",
        "run_dir",
        "config_path",
        "checkpoint_path",
        "note",
    ] + metric_columns + TRACKED_KEYS

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            metrics = run.get("metrics", {})
            config = run.get("config", {})
            row = {
                "run_id": run.get("run_id"),
                "stage": run.get("stage"),
                "label": run.get("label"),
                "status": run.get("status"),
                "objective_value": run.get("objective_value"),
                "objective_used": run.get("objective_used"),
                "runtime_sec": run.get("runtime_sec"),
                "exit_code": run.get("exit_code"),
                "run_dir": run.get("run_dir"),
                "config_path": run.get("config_path"),
                "checkpoint_path": run.get("checkpoint_path"),
                "note": run.get("note"),
            }
            for key in metric_columns:
                row[key] = metrics.get(key)
            for key in TRACKED_KEYS:
                row[key] = config.get(key)
            writer.writerow(row)


def best_completed_result(runs: Sequence[Dict[str, Any]], objective_key: str) -> Optional[Dict[str, Any]]:
    completed = [run for run in runs if run.get("status") == "completed" and run.get("objective_value") is not None]
    if not completed:
        return None

    def sort_key(item: Dict[str, Any]) -> tuple:
        metrics = item.get("metrics", {})
        return (
            float(item.get("objective_value", float("-inf"))),
            float(metrics.get("final_p_final_top1", float("-inf"))),
            -float(item.get("runtime_sec", 0.0)),
        )

    return max(completed, key=sort_key)


def derive_objective(metrics: Dict[str, Any], objective_key: str) -> tuple[Optional[float], Optional[str]]:
    for key in OBJECTIVE_FALLBACKS[objective_key]:
        value = metrics.get(key)
        if value is not None:
            return float(value), key
    return None, None


def prepare_base_config(raw_config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(raw_config)
    cfg["model_name"] = "spie"
    target_dataset = str(args.target_dataset).strip() if args.target_dataset else str(cfg.get("dataset", "")).strip()
    if target_dataset:
        cfg["dataset"] = target_dataset
    cfg["swanlab"] = True
    cfg["prefix"] = args.force_prefix
    return cfg


def apply_global_epoch_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    for key in (
        "task0_shared_epochs",
        "shared_cls_epochs",
        "task0_expert_epochs",
        "incremental_expert_epochs",
        "shared_cls_crct_epochs",
    ):
        cfg[key] = scaled_int(cfg[key], factor)
    return cfg


def apply_shared_epoch_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["task0_shared_epochs"] = scaled_int(cfg["task0_shared_epochs"], factor)
    cfg["shared_cls_epochs"] = scaled_int(cfg["shared_cls_epochs"], factor)
    return cfg


def apply_expert_epoch_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["task0_expert_epochs"] = scaled_int(cfg["task0_expert_epochs"], factor)
    cfg["incremental_expert_epochs"] = scaled_int(cfg["incremental_expert_epochs"], factor)
    return cfg


def apply_ca_epoch_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["shared_cls_crct_epochs"] = scaled_int(cfg["shared_cls_crct_epochs"], factor)
    return cfg


def apply_shared_lr_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["task0_shared_lr"] = scaled_float(cfg["task0_shared_lr"], factor)
    cfg["shared_cls_lr"] = scaled_float(cfg["shared_cls_lr"], factor)
    return cfg


def apply_expert_lr_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["task0_expert_lr"] = scaled_float(cfg["task0_expert_lr"], factor)
    cfg["incremental_expert_lr"] = scaled_float(cfg["incremental_expert_lr"], factor)
    return cfg


def apply_ca_lr_scale(config: Dict[str, Any], factor: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["shared_cls_ca_lr"] = scaled_float(cfg["shared_cls_ca_lr"], factor)
    return cfg


def set_distill_lambda(config: Dict[str, Any], value: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["expert_shape_distill_lambda"] = stable_float(value)
    return cfg


def set_distill_temperature(config: Dict[str, Any], value: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["expert_shape_distill_temperature"] = stable_float(value)
    return cfg


def set_cap_ratio(config: Dict[str, Any], value: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["expert_shape_reg_cap_ratio"] = stable_float(value)
    return cfg


def set_margin(config: Dict[str, Any], value: float) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg["m"] = stable_float(value)
    return cfg


def build_run_result(
    *,
    run_id: str,
    stage: str,
    label: str,
    note: str,
    fingerprint: str,
    config: Dict[str, Any],
    config_path: Path,
    launcher_log: Path,
    status: str,
    run_dir: Optional[str],
    checkpoint_path: Optional[str],
    metrics: Dict[str, Any],
    objective_key: str,
    exit_code: Optional[int],
    runtime_sec: float,
    parent_run_id: Optional[str],
) -> Dict[str, Any]:
    objective_value, objective_used = derive_objective(metrics, objective_key)
    return {
        "run_id": run_id,
        "stage": stage,
        "label": label,
        "note": note,
        "fingerprint": fingerprint,
        "status": status,
        "exit_code": exit_code,
        "runtime_sec": stable_float(runtime_sec, digits=4),
        "run_dir": run_dir,
        "config_path": str(config_path),
        "launcher_log": str(launcher_log),
        "checkpoint_path": checkpoint_path,
        "metrics": metrics,
        "objective_value": objective_value,
        "objective_used": objective_used,
        "parent_run_id": parent_run_id,
        "config": config,
    }


def parse_train_log_metrics(train_log: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not train_log.exists():
        return metrics

    with open(train_log, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = AVERAGE_ACCURACY_RE.search(line)
            if not match:
                continue
            raw_name = match.group(1)
            value = float(match.group(2))
            metrics[f"avg_{metric_name_to_key(raw_name)}_top1"] = value
    return metrics


def find_latest_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = []
    for path in ckpt_dir.glob("task_*.pkl"):
        match = TASK_CKPT_RE.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def parse_checkpoint_metrics(checkpoint_path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not checkpoint_path.exists():
        return metrics

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint.get("cnn_accy"), dict):
        metrics["final_cnn_top1"] = float(checkpoint["cnn_accy"].get("top1"))
    if isinstance(checkpoint.get("nme_accy"), dict):
        metrics["final_nme_top1"] = float(checkpoint["nme_accy"].get("top1"))
    if isinstance(checkpoint.get("eval_variants"), dict):
        for name, accy in checkpoint["eval_variants"].items():
            if isinstance(accy, dict) and "top1" in accy:
                metrics[f"final_{metric_name_to_key(name)}_top1"] = float(accy["top1"])
    return metrics


def merge_metrics(*metric_dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for metric_dict in metric_dicts:
        merged.update(metric_dict)
    return merged


def parse_run_dir_from_stream_line(line: str) -> Optional[str]:
    match = RUN_DIR_RE.search(line)
    if match:
        return match.group(1)
    return None


def fallback_find_run_dir(note: str) -> Optional[str]:
    candidates = sorted(
        (path for path in (REPO_ROOT / "logs").glob(f"*{sanitize_text(note)}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None
    return str(candidates[-1])


def run_training_subprocess(
    *,
    python_exe: str,
    config_path: Path,
    note: str,
    launcher_log: Path,
    gpu: Optional[str],
) -> tuple[int, Optional[str]]:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [python_exe, "-u", "main.py", "--config", str(config_path), "--note", note]
    run_dir: Optional[str] = None

    with open(launcher_log, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_f.write(line)
            if run_dir is None:
                parsed = parse_run_dir_from_stream_line(line)
                if parsed:
                    run_dir = parsed

        exit_code = process.wait()

    if run_dir is None:
        run_dir = fallback_find_run_dir(note)
    return exit_code, run_dir


def existing_result_for_fingerprint(
    runs: Sequence[Dict[str, Any]],
    fingerprint: str,
    rerun_failed: bool,
) -> Optional[Dict[str, Any]]:
    for run in runs:
        if run.get("fingerprint") != fingerprint:
            continue
        if run.get("status") == "failed" and rerun_failed:
            continue
        return run
    return None


def estimate_run_budget(preset_name: str) -> int:
    preset = SEARCH_PRESETS[preset_name]
    total = len(preset["global_epoch_scales"])
    total += len(preset["shared_epoch_scales"])
    total += len(preset["expert_epoch_scales"])
    total += len(preset["ca_epoch_scales"])
    total += len(preset["shared_lr_scales"])
    total += len(preset["expert_lr_scales"])
    total += len(preset["ca_lr_scales"])
    total += len(preset["lambda_values"])
    total += len(preset["temperature_values"])
    total += len(preset["cap_ratio_values"])
    total += len(preset["margin_values"])
    return total


class SweepRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.base_config_path = (REPO_ROOT / args.base_config).resolve()
        self.paths = init_output_dirs((REPO_ROOT / args.output_dir).resolve())
        self.state = load_state(self.paths["state"])
        self.stage_plan = SEARCH_PRESETS[args.preset]
        self.new_runs_launched = 0
        self.preview_counter = 0

        raw_base = load_json(self.base_config_path)
        self.base_config = prepare_base_config(raw_base, args)

        self.state.setdefault("base_config_path", str(self.base_config_path))
        self.state.setdefault("preset", args.preset)
        self.state.setdefault("objective", args.objective)
        self.state.setdefault("runs", [])

        if self.state["base_config_path"] != str(self.base_config_path):
            raise ValueError(
                f"Existing sweep state was created from {self.state['base_config_path']}, "
                f"not {self.base_config_path}."
            )

    def can_launch_more(self) -> bool:
        return self.args.max_runs is None or self.new_runs_launched < self.args.max_runs

    def evaluate_candidate(
        self,
        *,
        stage: str,
        label: str,
        config: Dict[str, Any],
        parent_run_id: Optional[str],
    ) -> Dict[str, Any]:
        fingerprint = config_fingerprint(config)
        existing = existing_result_for_fingerprint(
            self.state["runs"],
            fingerprint=fingerprint,
            rerun_failed=self.args.rerun_failed,
        )
        if existing is not None:
            print(
                f"[skip] {stage}/{label} already recorded with status={existing.get('status')} "
                f"objective={existing.get('objective_value')}"
            )
            return existing

        run_index = len(self.state["runs"]) + 1
        if self.args.dry_run:
            self.preview_counter += 1
            run_index = self.preview_counter
        run_id = f"run_{run_index:03d}"
        note = sanitize_text(f"{run_id}-{stage}-{label}")
        config_path = self.paths["configs"] / f"{run_id}_{sanitize_text(stage)}_{sanitize_text(label)}.json"
        launcher_log = self.paths["launcher_logs"] / f"{run_id}_{sanitize_text(stage)}_{sanitize_text(label)}.log"

        if self.args.dry_run:
            print(f"[dry-run] {run_id} {stage}/{label}")
            print(f"          objective={self.args.objective}")
            print(f"          tracked={{{', '.join(f'{k}={config.get(k)}' for k in TRACKED_KEYS if k in config)}}}")
            return build_run_result(
                run_id=run_id,
                stage=stage,
                label=label,
                note=note,
                fingerprint=fingerprint,
                config=config,
                config_path=config_path,
                launcher_log=launcher_log,
                status="dry_run",
                run_dir=None,
                checkpoint_path=None,
                metrics={},
                objective_key=self.args.objective,
                exit_code=None,
                runtime_sec=0.0,
                parent_run_id=parent_run_id,
            )

        if not self.can_launch_more():
            raise RuntimeError(f"Reached --max-runs={self.args.max_runs}.")

        write_json(config_path, config)

        print(f"[run] {run_id} {stage}/{label}")
        print(f"      config: {config_path}")
        start_time = time.time()
        exit_code, run_dir = run_training_subprocess(
            python_exe=self.args.python,
            config_path=config_path,
            note=note,
            launcher_log=launcher_log,
            gpu=self.args.gpu,
        )
        runtime_sec = time.time() - start_time
        self.new_runs_launched += 1

        metrics: Dict[str, Any] = {}
        checkpoint_path: Optional[str] = None
        status = "completed" if exit_code == 0 else "failed"

        if run_dir is not None:
            run_dir_path = Path(run_dir)
            train_log = run_dir_path / "train.log"
            metrics = merge_metrics(metrics, parse_train_log_metrics(train_log))
            latest_ckpt = find_latest_checkpoint(run_dir_path)
            if latest_ckpt is not None:
                checkpoint_path = str(latest_ckpt)
                metrics = merge_metrics(metrics, parse_checkpoint_metrics(latest_ckpt))

        result = build_run_result(
            run_id=run_id,
            stage=stage,
            label=label,
            note=note,
            fingerprint=fingerprint,
            config=config,
            config_path=config_path,
            launcher_log=launcher_log,
            status=status,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            metrics=metrics,
            objective_key=self.args.objective,
            exit_code=exit_code,
            runtime_sec=runtime_sec,
            parent_run_id=parent_run_id,
        )
        self.state["runs"].append(result)
        persist_state(self.paths, self.state, self.args.objective)

        print(
            f"[done] {run_id} status={status} objective={result.get('objective_value')} "
            f"used={result.get('objective_used')} runtime={runtime_sec / 60.0:.1f}m"
        )
        if status != "completed" and not self.args.continue_on_error:
            raise RuntimeError(f"Run {run_id} failed with exit code {exit_code}.")
        return result

    def choose_best(self, candidates: Sequence[Dict[str, Any]], fallback: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        completed = [
            candidate
            for candidate in candidates
            if candidate is not None
            and candidate.get("status") == "completed"
            and candidate.get("objective_value") is not None
        ]
        if completed:
            return best_completed_result(completed, self.args.objective)
        return fallback

    def run_global_epoch_stage(self) -> Optional[Dict[str, Any]]:
        results = []
        for scale in self.stage_plan["global_epoch_scales"]:
            config = apply_global_epoch_scale(self.base_config, scale)
            label = f"global_epoch_x{float_str(scale)}"
            results.append(
                self.evaluate_candidate(
                    stage="global_epoch",
                    label=label,
                    config=config,
                    parent_run_id=None,
                )
            )
        best = self.choose_best(results)
        if best is not None:
            return best
        if self.args.dry_run:
            for scale, result in zip(self.stage_plan["global_epoch_scales"], results):
                if stable_float(float(scale)) == 1.0:
                    return result
            return results[0] if results else None
        return None

    def run_coordinate_stage(
        self,
        *,
        current_best: Dict[str, Any],
        stage: str,
        label_prefix: str,
        values: Iterable[Any],
        apply_fn,
        current_value_getter,
    ) -> Dict[str, Any]:
        anchor = current_best
        stage_results = []
        anchor_value = current_value_getter(anchor["config"])
        for value in values:
            if stable_float(float(value)) == stable_float(float(anchor_value)):
                continue
            config = apply_fn(anchor["config"], value)
            label = f"{label_prefix}_{float_str(float(value))}"
            stage_results.append(
                self.evaluate_candidate(
                    stage=stage,
                    label=label,
                    config=config,
                    parent_run_id=anchor.get("run_id"),
                )
            )
        best = self.choose_best([anchor] + stage_results, fallback=anchor)
        assert best is not None
        return best

    def execute(self) -> Optional[Dict[str, Any]]:
        print(f"[info] Adaptive SPiE {self.base_config.get('dataset', 'dataset')} sweep")
        print(f"[info] base_config: {self.base_config_path}")
        print(f"[info] output_dir: {self.paths['root']}")
        print(f"[info] preset: {self.args.preset}")
        print(f"[info] objective: {self.args.objective}")
        print(f"[info] estimated upper-bound runs: {estimate_run_budget(self.args.preset)}")
        if self.args.max_runs is not None:
            print(f"[info] max newly launched runs this invocation: {self.args.max_runs}")
        if self.args.gpu is not None:
            print(f"[info] CUDA_VISIBLE_DEVICES={self.args.gpu}")
        print()

        best = self.run_global_epoch_stage()
        if best is None:
            return None
        print(f"[best] after global_epoch -> {best['run_id']} objective={best.get('objective_value')}")

        best = self.run_coordinate_stage(
            current_best=best,
            stage="shared_epoch",
            label_prefix="shared_epoch_x",
            values=self.stage_plan["shared_epoch_scales"],
            apply_fn=apply_shared_epoch_scale,
            current_value_getter=lambda cfg: 1.0,
        )
        print(f"[best] after shared_epoch -> {best['run_id']} objective={best.get('objective_value')}")

        best = self.run_coordinate_stage(
            current_best=best,
            stage="expert_epoch",
            label_prefix="expert_epoch_x",
            values=self.stage_plan["expert_epoch_scales"],
            apply_fn=apply_expert_epoch_scale,
            current_value_getter=lambda cfg: 1.0,
        )
        print(f"[best] after expert_epoch -> {best['run_id']} objective={best.get('objective_value')}")

        best = self.run_coordinate_stage(
            current_best=best,
            stage="ca_epoch",
            label_prefix="ca_epoch_x",
            values=self.stage_plan["ca_epoch_scales"],
            apply_fn=apply_ca_epoch_scale,
            current_value_getter=lambda cfg: 1.0,
        )
        print(f"[best] after ca_epoch -> {best['run_id']} objective={best.get('objective_value')}")

        if self.stage_plan["shared_lr_scales"]:
            best = self.run_coordinate_stage(
                current_best=best,
                stage="shared_lr",
                label_prefix="shared_lr_x",
                values=self.stage_plan["shared_lr_scales"],
                apply_fn=apply_shared_lr_scale,
                current_value_getter=lambda cfg: 1.0,
            )
            print(f"[best] after shared_lr -> {best['run_id']} objective={best.get('objective_value')}")

        if self.stage_plan["expert_lr_scales"]:
            best = self.run_coordinate_stage(
                current_best=best,
                stage="expert_lr",
                label_prefix="expert_lr_x",
                values=self.stage_plan["expert_lr_scales"],
                apply_fn=apply_expert_lr_scale,
                current_value_getter=lambda cfg: 1.0,
            )
            print(f"[best] after expert_lr -> {best['run_id']} objective={best.get('objective_value')}")

        if self.stage_plan["ca_lr_scales"]:
            best = self.run_coordinate_stage(
                current_best=best,
                stage="ca_lr",
                label_prefix="ca_lr_x",
                values=self.stage_plan["ca_lr_scales"],
                apply_fn=apply_ca_lr_scale,
                current_value_getter=lambda cfg: 1.0,
            )
            print(f"[best] after ca_lr -> {best['run_id']} objective={best.get('objective_value')}")

        best = self.run_coordinate_stage(
            current_best=best,
            stage="distill_lambda",
            label_prefix="lambda",
            values=self.stage_plan["lambda_values"],
            apply_fn=set_distill_lambda,
            current_value_getter=lambda cfg: cfg["expert_shape_distill_lambda"],
        )
        print(f"[best] after distill_lambda -> {best['run_id']} objective={best.get('objective_value')}")

        if self.stage_plan["temperature_values"]:
            best = self.run_coordinate_stage(
                current_best=best,
                stage="distill_temperature",
                label_prefix="temperature",
                values=self.stage_plan["temperature_values"],
                apply_fn=set_distill_temperature,
                current_value_getter=lambda cfg: cfg["expert_shape_distill_temperature"],
            )
            print(f"[best] after distill_temperature -> {best['run_id']} objective={best.get('objective_value')}")

        if self.stage_plan["cap_ratio_values"]:
            best = self.run_coordinate_stage(
                current_best=best,
                stage="distill_cap",
                label_prefix="cap",
                values=self.stage_plan["cap_ratio_values"],
                apply_fn=set_cap_ratio,
                current_value_getter=lambda cfg: cfg["expert_shape_reg_cap_ratio"],
            )
            print(f"[best] after distill_cap -> {best['run_id']} objective={best.get('objective_value')}")

        if self.stage_plan["margin_values"]:
            best = self.run_coordinate_stage(
                current_best=best,
                stage="margin",
                label_prefix="m",
                values=self.stage_plan["margin_values"],
                apply_fn=set_margin,
                current_value_getter=lambda cfg: cfg["m"],
            )
            print(f"[best] after margin -> {best['run_id']} objective={best.get('objective_value')}")

        persist_state(self.paths, self.state, self.args.objective)
        return best


def run_from_args(args: argparse.Namespace) -> None:
    runner = SweepRunner(args)
    best = runner.execute()

    if best is None:
        print("[warn] No completed run produced a ranked result.")
        return

    print("\n[summary] best result")
    print(f"  run_id: {best.get('run_id')}")
    print(f"  stage: {best.get('stage')}")
    print(f"  label: {best.get('label')}")
    print(f"  objective: {best.get('objective_used')} = {best.get('objective_value')}")
    print(f"  run_dir: {best.get('run_dir')}")
    print(f"  checkpoint: {best.get('checkpoint_path')}")
    print(f"  best_config: {runner.paths['best_config']}")
    print(f"  best_result: {runner.paths['best_result']}")
    print(f"  results_csv: {runner.paths['results_csv']}")
    print(f"  rerun_best: {runner.paths['rerun_best']}")


def main() -> None:
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
