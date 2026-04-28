#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=3

CONFIGS=(
    "exps/domainnet/tuna.json"
    "exps/omnibenchmark/dualprompt.json"
    "exps/domainnet/slca.json"
    "exps/omnibenchmark/coda_prompt.json"
    "exps/domainnet/ranpac.json"
    "exps/omnibenchmark/ease.json"
    "exps/domainnet/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU3] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu3"
done
