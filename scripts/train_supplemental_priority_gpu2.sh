#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=2
NOTE="${NOTE:-supplemental-priority-gpu2}"

CONFIGS=(
    # High priority: table has values but SwanLab has no record.
    "exps/cub/cofima.json"
    "exps/cub/dualprompt.json"

    # Medium priority: 20-step baselines.
    "exps/cifar224/coda_prompt-20step.json"
    "exps/cifar224/acil-20step.json"
    "exps/cifar224/ease-20step.json"
    "exps/cifar224/tuna-20step.json"
    "exps/cub/dualprompt-20step.json"
    "exps/cub/slca-20step.json"
    "exps/cub/fecam-20step.json"
    "exps/cub/mos-20step.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py --allow-20step "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU2] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
