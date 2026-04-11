#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  "exps/spie_v13_imga_grid_base.json"
  "exps/spie_v13_imga_grid_sharedcls_long.json"
  "exps/spie_v13_imga_grid_calib_soft.json"
  "exps/spie_v13_imga_grid_margin_high.json"
)

for config in "${CONFIGS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${config}"
  python main.py --config "$config"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${config}"
done
