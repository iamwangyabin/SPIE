#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=1
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu1}"

CONFIGS=(
    "exps/cifar224/spie.json"
    "exps/cub/l2p.json"
    "exps/imagenetr/fecam.json"
    "exps/cifar224/mos.json"
    "exps/cub/min.json"
    "exps/imagenetr/ssiat.json"
    "exps/cifar224/tuna.json"
    "exps/cub/dualprompt.json"
    "exps/imagenetr/coda_prompt.json"
    "exps/cifar224/ranpac.json"
    "exps/cub/ease.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU1] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
