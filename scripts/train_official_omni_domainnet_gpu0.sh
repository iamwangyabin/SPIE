#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0

CONFIGS=(
    "exps/omnibenchmark/official/spie.json"
    "exps/domainnet/official/l2p.json"
    "exps/omnibenchmark/official/acil.json"
    "exps/domainnet/official/fecam.json"
    "exps/omnibenchmark/official/mos.json"
    "exps/domainnet/official/min.json"
    "exps/omnibenchmark/official/aper.json"
    "exps/domainnet/official/ssiat.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU0] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu0"
done
