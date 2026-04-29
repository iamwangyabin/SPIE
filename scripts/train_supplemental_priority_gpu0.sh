#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0
NOTE="${NOTE:-supplemental-priority-gpu0}"

CONFIGS=(
    # High priority: table has values but SwanLab has no record.
    "exps/imagenetr/cofima.json"
    "exps/cub/ease.json"

    # Medium priority: 20-step baselines.
    "exps/cifar224/l2p-20step.json"
    "exps/cifar224/slca-20step.json"
    "exps/cifar224/ssiat-20step.json"
    "exps/cifar224/ranpac-20step.json"
    "exps/cub/acil-20step.json"
    "exps/cub/ease-20step.json"
    "exps/cub/min-20step.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py --allow-20step "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU0] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
