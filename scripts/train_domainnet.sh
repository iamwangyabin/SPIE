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
    run_config "exps/domainnet/tuna_domainnet_strong_10step.json" "bash-domainnet-tuna-strong-10step"
    run_config "exps/domainnet/tuna_domainnet_strong_20step.json" "bash-domainnet-tuna-strong-20step"
}

run_tuna_shuffle() {
    run_config "exps/domainnet/tuna_domainnet_normal_shuffle_10step.json" "bash-domainnet-tuna-normal-shuffle-10step"
    run_config "exps/domainnet/tuna_domainnet_normal_shuffle_20step.json" "bash-domainnet-tuna-normal-shuffle-20step"
}

run_tuna_easy() {
    run_config "exps/domainnet/tuna_domainnet_easy_10step.json" "bash-domainnet-tuna-easy-10step"
    run_config "exps/domainnet/tuna_domainnet_easy_20step.json" "bash-domainnet-tuna-easy-20step"
}

run_spie() {
    run_config "exps/domainnet/spie_domainnet_strong_10step.json" "bash-domainnet-spie-strong-10step"
    run_config "exps/domainnet/spie_domainnet_strong_20step.json" "bash-domainnet-spie-strong-20step"
}

run_spie_shuffle() {
    run_config "exps/domainnet/spie_domainnet_normal_shuffle_10step.json" "bash-domainnet-spie-normal-shuffle-10step"
    run_config "exps/domainnet/spie_domainnet_normal_shuffle_20step.json" "bash-domainnet-spie-normal-shuffle-20step"
}

run_spie_easy() {
    run_config "exps/domainnet/spie_domainnet_easy_10step.json" "bash-domainnet-spie-easy-10step"
    run_config "exps/domainnet/spie_domainnet_easy_20step.json" "bash-domainnet-spie-easy-20step"
}

run_spie_moretrain() {
    run_config "exps/domainnet/spie_domainnet_strong_10step_moretrain.json" "bash-domainnet-spie-strong-10step-moretrain"
    run_config "exps/domainnet/spie_domainnet_strong_20step_moretrain.json" "bash-domainnet-spie-strong-20step-moretrain"
}

case "${TARGET}" in
    tuna)
        run_tuna
        ;;
    tuna_shuffle)
        run_tuna_shuffle
        ;;
    tuna_easy)
        run_tuna_easy
        ;;
    spie)
        run_spie
        ;;
    spie_shuffle)
        run_spie_shuffle
        ;;
    spie_easy)
        run_spie_easy
        ;;
    spie_moretrain)
        run_spie_moretrain
        ;;
    shuffle)
        run_tuna_shuffle
        run_spie_shuffle
        ;;
    easy)
        run_tuna_easy
        run_spie_easy
        ;;
    all)
        run_tuna
        run_spie
        ;;
    *)
        echo "Usage: $0 [tuna|tuna_shuffle|tuna_easy|spie|spie_shuffle|spie_easy|spie_moretrain|shuffle|easy|all]" >&2
        exit 1
        ;;
esac
