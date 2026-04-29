#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NOTE="${NOTE:-official-imageneta-10step-all-methods}"

CONFIGS=(
    "exps/imageneta/acil.json"
    "exps/imageneta/aper.json"
    "exps/imageneta/coda_prompt.json"
    "exps/imageneta/cofima.json"
    "exps/imageneta/dualprompt.json"
    "exps/imageneta/ease.json"
    "exps/imageneta/fecam.json"
    "exps/imageneta/l2p.json"
    "exps/imageneta/min.json"
    "exps/imageneta/mos.json"
    "exps/imageneta/ranpac.json"
    "exps/imageneta/slca.json"
    "exps/imageneta/ssiat.json"
    "exps/imageneta/tuna.json"
    "exps/imageneta/spie.json"
)

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[ImageNet-A 10-step] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
