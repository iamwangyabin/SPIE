#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET="${1:-all}"

run_config() {
    local config_path="$1"
    local note="$2"
    local full_config="${ROOT_DIR}/${config_path}"

    if [[ ! -f "${full_config}" ]]; then
        echo "[ERROR] Config not found: ${config_path}" >&2
        exit 1
    fi

    echo "[RUN] ${config_path}"
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py" --config "${full_config}" --note "${note}"
}

run_tuna() {
    run_config "exps/tuna_domainnet_strong_10step.json" "bash-domainnet-tuna-strong-10step"
    run_config "exps/tuna_domainnet_strong_20step.json" "bash-domainnet-tuna-strong-20step"
}

run_spie() {
    run_config "exps/spie_domainnet_strong_10step.json" "bash-domainnet-spie-strong-10step"
    run_config "exps/spie_domainnet_strong_20step.json" "bash-domainnet-spie-strong-20step"
}

run_spie_moretrain() {
    run_config "exps/spie_domainnet_strong_10step_moretrain.json" "bash-domainnet-spie-strong-10step-moretrain"
    run_config "exps/spie_domainnet_strong_20step_moretrain.json" "bash-domainnet-spie-strong-20step-moretrain"
}

case "${TARGET}" in
    tuna)
        run_tuna
        ;;
    spie)
        run_spie
        ;;
    spie_moretrain)
        run_spie_moretrain
        ;;
    all)
        run_tuna
        run_spie
        ;;
    *)
        echo "Usage: $0 [tuna|spie|spie_moretrain|all]" >&2
        exit 1
        ;;
esac
