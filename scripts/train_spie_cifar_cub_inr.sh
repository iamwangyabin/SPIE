#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/train_spie_cifar.sh" "${1:-all}"
"${ROOT_DIR}/scripts/train_spie_cub.sh" "${1:-all}"
"${ROOT_DIR}/scripts/train_spie_inr.sh" "${1:-all}"
