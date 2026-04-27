#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=1

CONFIGS=(
    "exps/domainnet/official/spie.json"
    "exps/omnibenchmark/official/l2p.json"
    "exps/domainnet/official/acil.json"
    "exps/omnibenchmark/official/fecam.json"
    "exps/domainnet/official/mos.json"
    "exps/omnibenchmark/official/min.json"
    "exps/domainnet/official/aper.json"
    "exps/omnibenchmark/official/ssiat.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU1] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu1"
done
