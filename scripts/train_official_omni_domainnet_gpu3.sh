#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=3

CONFIGS=(
    "exps/domainnet/official/tuna.json"
    "exps/omnibenchmark/official/dualprompt.json"
    "exps/domainnet/official/slca.json"
    "exps/omnibenchmark/official/coda_prompt.json"
    "exps/domainnet/official/ranpac.json"
    "exps/omnibenchmark/official/ease.json"
    "exps/domainnet/official/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU3] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu3"
done
