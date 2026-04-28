#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=2
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu2}"

CONFIGS=(
    "exps/cub/spie.json"
    "exps/imagenetr/acil.json"
    "exps/cifar224/fecam.json"
    "exps/cub/mos.json"
    "exps/imagenetr/aper.json"
    "exps/cifar224/ssiat.json"
    "exps/cub/tuna.json"
    "exps/imagenetr/slca.json"
    "exps/cifar224/coda_prompt.json"
    "exps/cub/ranpac.json"
    "exps/imagenetr/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU2] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
