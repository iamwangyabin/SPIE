#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIGS=(
  "exps/spie_v13_inr_grid_base.json"
  "exps/spie_v13_inr_grid_temp_low.json"
  "exps/spie_v13_inr_grid_shared_low.json"
  "exps/spie_v13_inr_grid_calib_long.json"
)

for config in "${CONFIGS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${config}"
  python main.py --config "$config"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${config}"
done
