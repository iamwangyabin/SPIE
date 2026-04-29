#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=1
NOTE="${NOTE:-supplemental-priority-gpu1}"

CONFIGS=(
    # High priority: table has values but SwanLab has no record.
    "exps/cifar224/cofima.json"
    "exps/cub/coda_prompt.json"

    # Medium priority: 20-step baselines.
    "exps/cifar224/dualprompt-20step.json"
    "exps/cifar224/fecam-20step.json"
    "exps/cifar224/aper-20step.json"
    "exps/cifar224/mos-20step.json"
    "exps/cub/l2p-20step.json"
    "exps/cub/ssiat-20step.json"
    "exps/cub/tuna-20step.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py --allow-20step "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU1] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
