#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=2

CONFIGS=(
    "exps/omnibenchmark/tuna.json"
    "exps/domainnet/dualprompt.json"
    "exps/omnibenchmark/slca.json"
    "exps/domainnet/coda_prompt.json"
    "exps/omnibenchmark/ranpac.json"
    "exps/domainnet/ease.json"
    "exps/omnibenchmark/cofima.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU2] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu2"
done
