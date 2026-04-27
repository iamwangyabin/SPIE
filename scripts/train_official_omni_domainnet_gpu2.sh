#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=2

CONFIGS=(
    "exps/omnibenchmark/official/tuna.json"
    "exps/domainnet/official/dualprompt.json"
    "exps/omnibenchmark/official/slca.json"
    "exps/domainnet/official/coda_prompt.json"
    "exps/omnibenchmark/official/ranpac.json"
    "exps/domainnet/official/ease.json"
    "exps/omnibenchmark/official/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU2] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu2"
done
