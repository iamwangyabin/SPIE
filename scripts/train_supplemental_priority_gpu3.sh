#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=3
NOTE="${NOTE:-supplemental-priority-gpu3}"

CONFIGS=(
    # High priority: table has values but SwanLab has no record.
    "exps/imagenetr/ease.json"
    "exps/cub/ranpac.json"

    # Medium priority: TUNA CIFAR-100 10-step backfill, then 20-step baselines.
    "exps/cifar224/tuna.json"
    "exps/cifar224/cofima-20step.json"
    "exps/cifar224/min-20step.json"
    "exps/cub/coda_prompt-20step.json"
    "exps/cub/ranpac-20step.json"
    "exps/cub/aper-20step.json"
    "exps/cub/cofima-20step.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py --allow-20step "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU3] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
