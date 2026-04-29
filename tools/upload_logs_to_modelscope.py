#!/usr/bin/env python3
"""
Upload completed experiment log directories to a ModelScope model repository.

Usage:
  export MODELSCOPE_API_TOKEN=your_modelscope_token
  python tools/upload_logs_to_modelscope.py
  python tools/upload_logs_to_modelscope.py --logs-dir logs/<run_name>
"""

import argparse
import ast
import math
import re
import shutil
from pathlib import Path

CHECKPOINT_PATTERN = re.compile(r"task_(\d+)\.pkl$")
DEFAULT_LOGS_DIR_NAME = "logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload completed experiment log directories to ModelScope."
    )
    parser.add_argument(
        "--repo-id",
        default="yabinnng/SPIE",
        help="ModelScope repo id, e.g. username/repo_name.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help=(
            "Local logs root or a specific run directory to upload. "
            "If this points to logs/, every completed child run directory is uploaded."
        ),
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help=(
            "Target directory inside the ModelScope repo. "
            "If omitted, use <run_dir_name> to avoid overwriting other runs."
        ),
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="Keep the local run directory after a successful upload.",
    )
    parser.add_argument(
        "--commit-message",
        default="upload logs",
        help="Commit message on ModelScope.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="ModelScope token. If omitted, use MODELSCOPE_API_TOKEN or cached login.",
    )
    return parser.parse_args()


def find_run_dirs(logs_root):
    run_dirs = [
        path
        for path in logs_root.iterdir()
        if path.is_dir() and any(child.is_file() for child in path.rglob("*"))
    ]
    return sorted(run_dirs, key=lambda path: path.stat().st_mtime)


def parse_int_arg_from_log(log_path, arg_name):
    pattern = re.compile(rf"=>\s*{re.escape(arg_name)}:\s*(-?\d+)\b")
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None


def parse_class_count_from_log(log_path):
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if "[data_manager.py]" not in line or "=>" not in line:
                continue
            payload = line.split("=>", 1)[1].strip()
            if not payload.startswith("["):
                continue
            try:
                class_order = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue
            if isinstance(class_order, list) and all(
                isinstance(item, int) for item in class_order
            ):
                return len(class_order)
    return None


def infer_expected_task_count(run_dir):
    log_path = run_dir / "train.log"
    if not log_path.is_file():
        return None, "missing train.log"

    init_cls = parse_int_arg_from_log(log_path, "init_cls")
    increment = parse_int_arg_from_log(log_path, "increment")
    nb_classes = parse_class_count_from_log(log_path)
    if init_cls is None or increment is None or nb_classes is None:
        return None, "cannot infer init_cls, increment, or class count from train.log"
    if init_cls <= 0 or increment <= 0 or nb_classes <= 0:
        return None, "invalid init_cls, increment, or class count in train.log"

    remaining_classes = max(0, nb_classes - init_cls)
    return 1 + math.ceil(remaining_classes / increment), None


def checkpoint_task_ids(run_dir):
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return set()

    task_ids = set()
    for checkpoint_path in checkpoint_dir.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
        if match and checkpoint_path.is_file():
            task_ids.add(int(match.group(1)))
    return task_ids


def completion_status(run_dir):
    expected_tasks, reason = infer_expected_task_count(run_dir)
    if expected_tasks is None:
        return False, reason

    task_ids = checkpoint_task_ids(run_dir)
    expected_ids = set(range(expected_tasks))
    missing_ids = sorted(expected_ids - task_ids)
    if missing_ids:
        preview = ",".join(str(task_id) for task_id in missing_ids[:5])
        if len(missing_ids) > 5:
            preview += ",..."
        return False, f"missing checkpoint task ids: {preview}"

    return True, f"{len(task_ids)}/{expected_tasks} checkpoints"


def is_completed_run_dir(run_dir):
    completed, _ = completion_status(run_dir)
    return completed


def completed_upload_target(run_dir, path_in_repo):
    completed, reason = completion_status(run_dir)
    if not completed:
        print(f"Skipping incomplete run directory: {run_dir} ({reason})")
        return None
    return run_dir, path_in_repo


def resolve_upload_targets(logs_dir, path_in_repo):
    if path_in_repo:
        normalized_path = path_in_repo.strip("/")
        if not normalized_path:
            raise ValueError("--path-in-repo must not be empty.")
        if logs_dir.name == DEFAULT_LOGS_DIR_NAME:
            raise ValueError(
                "--path-in-repo can only be used with a specific run directory. "
                "Pass --logs-dir logs/<run_name>."
            )
        target = completed_upload_target(logs_dir, normalized_path)
        return [target] if target is not None else []

    if logs_dir.name == DEFAULT_LOGS_DIR_NAME:
        targets = []
        for run_dir in find_run_dirs(logs_dir):
            target = completed_upload_target(run_dir, run_dir.name)
            if target is not None:
                targets.append(target)
        return targets

    target = completed_upload_target(logs_dir, logs_dir.name)
    return [target] if target is not None else []


def delete_uploaded_dir(upload_dir):
    resolved_upload_dir = upload_dir.resolve()
    resolved_cwd = Path.cwd().resolve()
    resolved_home = Path.home().resolve()
    resolved_root = Path(resolved_upload_dir.anchor).resolve()

    protected_dirs = {
        resolved_root,
        resolved_cwd,
        resolved_home,
        resolved_cwd.parent,
        resolved_home.parent,
    }
    if resolved_upload_dir in protected_dirs:
        raise ValueError(f"Refusing to delete protected directory: {upload_dir}")

    shutil.rmtree(upload_dir)


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir).expanduser()
    if not logs_dir.exists() or not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    upload_targets = resolve_upload_targets(logs_dir, args.path_in_repo)
    if not upload_targets:
        print(f"No completed run directories found under {logs_dir}; nothing to upload.")
        return

    from modelscope.hub.api import HubApi

    api = HubApi(token=args.token)
    api.login(args.token)

    for upload_dir, path_in_repo in upload_targets:
        files = [p for p in upload_dir.rglob("*") if p.is_file()]
        if not files:
            print(f"No files found under {upload_dir}; skipping.")
            continue

        print(f"Uploading {len(files)} files from {upload_dir} to {args.repo_id}/{path_in_repo}")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(upload_dir),
            path_in_repo=path_in_repo,
            commit_message=args.commit_message,
            token=args.token,
        )

        print(f"Upload committed to {args.repo_id}/{path_in_repo}.")
        if args.keep_local:
            print(f"Kept local run directory: {upload_dir}")
        else:
            delete_uploaded_dir(upload_dir)
            print(f"Deleted local run directory: {upload_dir}")


if __name__ == "__main__":
    main()
