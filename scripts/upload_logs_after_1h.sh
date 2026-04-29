#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DELAY_SECONDS="${DELAY_SECONDS:-3600}"
LOGS_DIR="${LOGS_DIR:-logs}"
REPO_ID="${REPO_ID:-yabinnng/SPIE}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-upload logs}"

ARGS=(
    --repo-id "${REPO_ID}"
    --logs-dir "${LOGS_DIR}"
    --commit-message "${COMMIT_MESSAGE}"
)

if [[ -n "${PATH_IN_REPO:-}" ]]; then
    ARGS+=(--path-in-repo "${PATH_IN_REPO}")
fi

if [[ "${KEEP_LOCAL:-0}" == "1" ]]; then
    ARGS+=(--keep-local)
fi

if [[ -n "${MODELSCOPE_API_TOKEN:-}" ]]; then
    ARGS+=(--token "${MODELSCOPE_API_TOKEN}")
fi

cd "${ROOT_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting ${DELAY_SECONDS}s before uploading logs."
sleep "${DELAY_SECONDS}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting upload: tools/upload_logs_to_modelscope.py ${ARGS[*]}"
"${PYTHON_BIN}" tools/upload_logs_to_modelscope.py "${ARGS[@]}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Upload job finished."
