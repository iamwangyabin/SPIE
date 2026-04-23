#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEP="${1:-all}"

run_config() {
    local config_path="$1"
    local note="$2"
    echo "[RUN] ${config_path}"
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py" --config "${ROOT_DIR}/${config_path}" --note "${note}"
}

case "${STEP}" in
    10)
        run_config "exps/spie_inr_10step.json" "bash-inr-10step"
        ;;
    20)
        run_config "exps/spie_inr_20step.json" "bash-inr-20step"
        ;;
    50)
        run_config "exps/spie_inr_50step.json" "bash-inr-50step"
        ;;
    all)
        run_config "exps/spie_inr_10step.json" "bash-inr-10step"
        run_config "exps/spie_inr_20step.json" "bash-inr-20step"
        run_config "exps/spie_inr_50step.json" "bash-inr-50step"
        ;;
    *)
        echo "Usage: $0 [10|20|50|all]" >&2
        exit 1
        ;;
esac
