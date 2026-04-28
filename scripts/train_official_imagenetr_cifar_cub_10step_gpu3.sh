#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=3
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu3}"

CONFIGS=(
    "exps/imagenetr/l2p.json"
    "exps/cifar224/acil.json"
    "exps/cub/fecam.json"
    "exps/imagenetr/min.json"
    "exps/cifar224/aper.json"
    "exps/cub/ssiat.json"
    "exps/imagenetr/dualprompt.json"
    "exps/cifar224/slca.json"
    "exps/cub/coda_prompt.json"
    "exps/imagenetr/ease.json"
    "exps/cifar224/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU3] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
