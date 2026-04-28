#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0
NOTE="${NOTE:-official-imagenetr-cifar-cub-10step-gpu0}"

CONFIGS=(
    "exps/imagenetr/official/spie.json"
    "exps/cifar224/official/l2p.json"
    "exps/cub/official/acil.json"
    "exps/imagenetr/official/mos.json"
    "exps/cifar224/official/min.json"
    "exps/cub/official/aper.json"
    "exps/imagenetr/official/tuna.json"
    "exps/cifar224/official/dualprompt.json"
    "exps/cub/official/slca.json"
    "exps/imagenetr/official/ranpac.json"
    "exps/cifar224/official/ease.json"
    "exps/cub/official/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/build_10step_comparison_configs.py
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU0] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
