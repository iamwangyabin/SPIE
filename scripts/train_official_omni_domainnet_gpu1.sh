#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES=1

CONFIGS=(
    "exps/domainnet/spie.json"
    "exps/omnibenchmark/l2p.json"
    "exps/domainnet/acil.json"
    "exps/omnibenchmark/fecam.json"
    "exps/domainnet/mos.json"
    "exps/omnibenchmark/min.json"
    "exps/domainnet/aper.json"
    "exps/omnibenchmark/ssiat.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[GPU1] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "official-omni-domainnet-gpu1"
done
