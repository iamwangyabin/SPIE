#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NOTE="${NOTE:-spie-lowrank16}"

CONFIGS=(
    "exps/ablations/spie_lowrank/domainnet/spie_domainnet_official_lowrank16.json"
    "exps/ablations/spie_lowrank/omnibenchmark/spie_omnibenchmark_10step_lowrank16.json"
    "exps/ablations/spie_lowrank/imagenetr/spie_inr_10step_lowrank16.json"
    "exps/ablations/spie_lowrank/cifar224/spie_cifar_10step_lowrank16.json"
    "exps/ablations/spie_lowrank/cifar224/spie_cifar_20step_lowrank16.json"
    "exps/ablations/spie_lowrank/cub/spie_cub_10step_lowrank16.json"
    "exps/ablations/spie_lowrank/cub/spie_cub_20step_lowrank16.json"
)

cd "${ROOT_DIR}"

for config in "${CONFIGS[@]}"; do
    if [[ ! -f "${config}" ]]; then
        echo "Missing config: ${config}" >&2
        exit 1
    fi
done

for config in "${CONFIGS[@]}"; do
    echo "[SPIE lowrank16][CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
