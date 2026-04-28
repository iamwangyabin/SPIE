#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=3
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu3}"

CONFIGS=(
    "exps/imagenetr/official/l2p.json"
    "exps/cifar224/official/acil.json"
    "exps/cub/official/fecam.json"
    "exps/imagenetr/official/min.json"
    "exps/cifar224/official/aper.json"
    "exps/cub/official/ssiat.json"
    "exps/imagenetr/official/dualprompt.json"
    "exps/cifar224/official/slca.json"
    "exps/cub/official/coda_prompt.json"
    "exps/imagenetr/official/ease.json"
    "exps/cifar224/official/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU3] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
