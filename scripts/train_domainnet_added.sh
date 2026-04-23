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

run_pair() {
    local stem="$1"
    local note_prefix="$2"
    run_config "exps/${stem}_10step.json" "${note_prefix}-10step"
    run_config "exps/${stem}_20step.json" "${note_prefix}-20step"
}

run_l2p() {
    run_pair "l2p_domainnet_strong" "bash-domainnet-l2p-strong"
}

run_dualprompt() {
    run_pair "dualprompt_domainnet_strong" "bash-domainnet-dualprompt-strong"
}

run_coda_prompt() {
    run_pair "coda_prompt_domainnet_strong" "bash-domainnet-coda-prompt-strong"
}

run_ease() {
    run_pair "ease_domainnet_strong" "bash-domainnet-ease-strong"
}

run_slca() {
    run_pair "slca_domainnet_strong" "bash-domainnet-slca-strong"
}

run_ranpac() {
    run_pair "ranpac_domainnet_strong" "bash-domainnet-ranpac-strong"
}

run_fecam() {
    run_pair "fecam_domainnet_strong" "bash-domainnet-fecam-strong"
}

run_aper_adapter() {
    run_pair "aper_adapter_domainnet_strong" "bash-domainnet-aper-adapter-strong"
}

run_aper() {
    run_aper_adapter
}

run_cofima() {
    run_pair "cofima_domainnet_strong" "bash-domainnet-cofima-strong"
}

run_mos() {
    run_pair "mos_domainnet_strong" "bash-domainnet-mos-strong"
}

run_all_added() {
    run_l2p
    run_dualprompt
    run_coda_prompt
    run_ease
    run_slca
    run_ranpac
    run_fecam
    run_aper
    run_cofima
    run_mos
}

case "${TARGET}" in
    l2p)
        run_l2p
        ;;
    dualprompt)
        run_dualprompt
        ;;
    coda|coda_prompt)
        run_coda_prompt
        ;;
    ease)
        run_ease
        ;;
    slca)
        run_slca
        ;;
    ranpac)
        run_ranpac
        ;;
    fecam)
        run_fecam
        ;;
    aper|aper_adapter)
        run_aper_adapter
        ;;
    cofima)
        run_cofima
        ;;
    mos)
        run_mos
        ;;
    all)
        run_all_added
        ;;
    *)
        echo "Usage: $0 [l2p|dualprompt|coda_prompt|ease|slca|ranpac|fecam|aper|aper_adapter|cofima|mos|all]" >&2
        exit 1
        ;;
esac
