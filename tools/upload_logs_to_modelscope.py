#!/usr/bin/env python3
"""
Upload the whole logs directory to a ModelScope model repository.

Usage:
  export MODELSCOPE_API_TOKEN=your_modelscope_token
  python tools/upload_logs_to_modelscope.py
"""

import argparse
from pathlib import Path

from modelscope.hub.api import HubApi


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload all files under logs/ to ModelScope."
    )
    parser.add_argument(
        "--repo-id",
        default="yabinnng/SPIE",
        help="ModelScope repo id, e.g. username/repo_name.",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Local logs directory to upload.",
    )
    parser.add_argument(
        "--path-in-repo",
        default="logs",
        help="Target directory inside the ModelScope repo.",
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


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists() or not logs_dir.is_dir():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    files = [p for p in logs_dir.rglob("*") if p.is_file()]
    if not files:
        print(f"No files found under {logs_dir}; nothing to upload.")
        return

    print(f"Uploading {len(files)} files from {logs_dir} to {args.repo_id}/{args.path_in_repo}")

    api = HubApi(token=args.token)
    api.login(args.token)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(logs_dir),
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        token=args.token,
    )

    print("Upload finished.")


if __name__ == "__main__":
    main()
