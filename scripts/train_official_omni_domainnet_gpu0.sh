#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=0

CONFIGS=(
    "exps/omnibenchmark/spie.json"
    "exps/domainnet/l2p.json"
    "exps/omnibenchmark/acil.json"
    "exps/domainnet/fecam.json"
    "exps/omnibenchmark/mos.json"
    "exps/domainnet/min.json"
    "exps/omnibenchmark/aper.json"
    "exps/domainnet/ssiat.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU0] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu0"
done
