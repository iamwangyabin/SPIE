#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu0}"

CONFIGS=(
    "exps/imagenetr/spie.json"
    "exps/cifar224/l2p.json"
    "exps/cub/acil.json"
    "exps/imagenetr/mos.json"
    "exps/cifar224/min.json"
    "exps/cub/aper.json"
    "exps/imagenetr/tuna.json"
    "exps/cifar224/dualprompt.json"
    "exps/cub/slca.json"
    "exps/imagenetr/ranpac.json"
    "exps/cifar224/ease.json"
    "exps/cub/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU0] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
