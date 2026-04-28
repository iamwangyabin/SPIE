#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=2
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu2}"

CONFIGS=(
    "exps/cub/official/spie.json"
    "exps/imagenetr/official/acil.json"
    "exps/cifar224/official/fecam.json"
    "exps/cub/official/mos.json"
    "exps/imagenetr/official/aper.json"
    "exps/cifar224/official/ssiat.json"
    "exps/cub/official/tuna.json"
    "exps/imagenetr/official/slca.json"
    "exps/cifar224/official/coda_prompt.json"
    "exps/cub/official/ranpac.json"
    "exps/imagenetr/official/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU2] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
