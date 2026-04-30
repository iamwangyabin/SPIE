#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
GROUP="${1:-all}"
NOTE="${NOTE:-spie-ablation}"
CONFIG_ROOT="${ROOT_DIR}/exps/imagenetr/spie_ablation"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-SPIE-ablation}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/generate_spie_ablation_configs.py

if [[ "${GROUP}" == "all" ]]; then
    mapfile -t CONFIGS < "${CONFIG_ROOT}/manifest.txt"
else
    if [[ ! -d "${CONFIG_ROOT}/${GROUP}" ]]; then
        echo "Unknown ablation group: ${GROUP}" >&2
        echo "Available groups: all $(find "${CONFIG_ROOT}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \\; | sort | tr '\n' ' ')" >&2
        exit 2
    fi
    mapfile -t CONFIGS < <(find "exps/imagenetr/spie_ablation/${GROUP}" -name "*.json" | sort)
fi

"${PYTHON_BIN}" tools/validate_official_configs.py "${CONFIGS[@]}"
"${PYTHON_BIN}" -c 'import json, pathlib, sys
bad = []
for item in sys.argv[1:]:
    cfg = json.loads(pathlib.Path(item).read_text())
    if cfg.get("swanlab") is not True or cfg.get("swanlab_mode") != "online":
        bad.append(item)
if bad:
    raise SystemExit("SwanLab online logging is not enabled for: " + ", ".join(bad))
print(f"SwanLab online logging enabled for {len(sys.argv) - 1} config(s).")
' "${CONFIGS[@]}"

for config in "${CONFIGS[@]}"; do
    echo "[SPIE ablation][GPU ${GPU_ID}] ${config}"
    "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
done
