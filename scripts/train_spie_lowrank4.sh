#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NOTE="${NOTE:-spie-lowrank4}"

CONFIGS=(
    "exps/domainnet/spie_domainnet_official_lowrank4.json"
    "exps/omnibenchmark/spie_omnibenchmark_10step_lowrank4.json"
    "exps/imagenetr/spie_inr_10step_lowrank4.json"
    "exps/cifar224/spie_cifar_10step_lowrank4.json"
    "exps/cifar224/spie_cifar_20step_lowrank4.json"
    "exps/cub/spie_cub_10step_lowrank4.json"
    "exps/cub/spie_cub_20step_lowrank4.json"
)

cd "${ROOT_DIR}"

for config in "${CONFIGS[@]}"; do
    if [[ ! -f "${config}" ]]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi
done

for config in "${CONFIGS[@]}"; do
    echo "[SPIE lowrank4][CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
