#!/usr/bin/env bash
set -euo pipefail

echo "[ERROR] This script has been superseded. Use scripts/train_official_omni_domainnet_gpu{0,1,2,3}.sh for OmniBenchmark 10-step and DomainNet official." >&2
exit 1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${1:-all}"
METHOD="${2:-all}"

normalize_dataset() {
    case "$1" in
        imagenetr|inr|image-net-r)
            printf '%s\n' "imagenetr"
            ;;
        omnibenchmark|omni)
            printf '%s\n' "omnibenchmark"
            ;;
        domainnet|domain)
            printf '%s\n' "domainnet"
            ;;
        all)
            printf '%s\n' "all"
            ;;
        *)
            echo "[ERROR] Unknown dataset: $1" >&2
            exit 1
            ;;
    esac
}

expand_method() {
    case "$1" in
        all)
            printf '%s\n' \
                l2p dualprompt coda_prompt slca ssiat fecam ranpac \
                aper_adapter aper_finetune aper_ssf aper_vpt_deep aper_vpt_shallow \
                ease cofima mos tuna min
            ;;
        coda)
            printf '%s\n' "coda_prompt"
            ;;
        aper)
            printf '%s\n' aper_adapter aper_finetune aper_ssf aper_vpt_deep aper_vpt_shallow
            ;;
        l2p|dualprompt|coda_prompt|slca|ssiat|fecam|ranpac|aper_adapter|aper_finetune|aper_ssf|aper_vpt_deep|aper_vpt_shallow|ease|cofima|mos|tuna|min)
            printf '%s\n' "$1"
            ;;
        acil)
            echo "[ERROR] ACIL is not registered in utils/factory.py, so there is no runnable config." >&2
            exit 1
            ;;
        *)
            echo "[ERROR] Unknown method: $1" >&2
            exit 1
            ;;
    esac
}

config_for() {
    local dataset="$1"
    local method="$2"

    case "${dataset}:${method}" in
        imagenetr:l2p) printf '%s\n' "exps/imagenetr/l2p_inr.json" ;;
        imagenetr:dualprompt) printf '%s\n' "exps/imagenetr/dualprompt_inr.json" ;;
        imagenetr:coda_prompt) printf '%s\n' "exps/imagenetr/coda_prompt_inr.json" ;;
        imagenetr:slca) printf '%s\n' "exps/imagenetr/slca_inr.json" ;;
        imagenetr:ssiat) printf '%s\n' "exps/imagenetr/ssiat_imagenetr.json" ;;
        imagenetr:fecam) printf '%s\n' "exps/imagenetr/fecam_inr.json" ;;
        imagenetr:ranpac) printf '%s\n' "exps/imagenetr/ranpac_inr.json" ;;
        imagenetr:aper_adapter) printf '%s\n' "exps/imagenetr/aper_adapter_inr.json" ;;
        imagenetr:aper_finetune) printf '%s\n' "exps/imagenetr/aper_finetune_inr.json" ;;
        imagenetr:aper_ssf) printf '%s\n' "exps/imagenetr/aper_ssf_inr.json" ;;
        imagenetr:aper_vpt_deep) printf '%s\n' "exps/imagenetr/aper_vpt_deep_inr.json" ;;
        imagenetr:aper_vpt_shallow) printf '%s\n' "exps/imagenetr/aper_vpt_shallow_inr.json" ;;
        imagenetr:ease) printf '%s\n' "exps/imagenetr/ease_inr.json" ;;
        imagenetr:cofima) printf '%s\n' "exps/imagenetr/cofima_inr.json" ;;
        imagenetr:mos) printf '%s\n' "exps/imagenetr/mos_inr.json" ;;
        imagenetr:tuna) printf '%s\n' "exps/imagenetr/tuna_inr_10step.json" ;;
        imagenetr:min) printf '%s\n' "exps/imagenetr/min_inr.json" ;;

        omnibenchmark:l2p) printf '%s\n' "exps/omnibenchmark/l2p_omnibenchmark_10step.json" ;;
        omnibenchmark:dualprompt) printf '%s\n' "exps/omnibenchmark/dualprompt_omnibenchmark_10step.json" ;;
        omnibenchmark:coda_prompt) printf '%s\n' "exps/omnibenchmark/coda_prompt_omnibenchmark_10step.json" ;;
        omnibenchmark:slca) printf '%s\n' "exps/omnibenchmark/slca_omnibenchmark_10step.json" ;;
        omnibenchmark:ssiat) printf '%s\n' "exps/omnibenchmark/ssiat_omnibenchmark_10step.json" ;;
        omnibenchmark:fecam) printf '%s\n' "exps/omnibenchmark/fecam_omnibenchmark_10step.json" ;;
        omnibenchmark:ranpac) printf '%s\n' "exps/omnibenchmark/ranpac_omnibenchmark_10step.json" ;;
        omnibenchmark:aper_adapter) printf '%s\n' "exps/omnibenchmark/aper_adapter_omnibenchmark_10step.json" ;;
        omnibenchmark:aper_finetune) printf '%s\n' "exps/omnibenchmark/aper_finetune_omnibenchmark_10step.json" ;;
        omnibenchmark:aper_ssf) printf '%s\n' "exps/omnibenchmark/aper_ssf_omnibenchmark_10step.json" ;;
        omnibenchmark:aper_vpt_deep) printf '%s\n' "exps/omnibenchmark/aper_vpt_deep_omnibenchmark_10step.json" ;;
        omnibenchmark:aper_vpt_shallow) printf '%s\n' "exps/omnibenchmark/aper_vpt_shallow_omnibenchmark_10step.json" ;;
        omnibenchmark:ease) printf '%s\n' "exps/omnibenchmark/ease_omnibenchmark_10step.json" ;;
        omnibenchmark:cofima) printf '%s\n' "exps/omnibenchmark/cofima_omnibenchmark_10step.json" ;;
        omnibenchmark:mos) printf '%s\n' "exps/omnibenchmark/mos_omnibenchmark_10step.json" ;;
        omnibenchmark:tuna) printf '%s\n' "exps/omnibenchmark/tuna_omnibenchmark_10step.json" ;;
        omnibenchmark:min) printf '%s\n' "exps/omnibenchmark/min_omnibenchmark_10step.json" ;;

        domainnet:l2p) printf '%s\n' "exps/domainnet/l2p_domainnet_official.json" ;;
        domainnet:dualprompt) printf '%s\n' "exps/domainnet/dualprompt_domainnet_official.json" ;;
        domainnet:coda_prompt) printf '%s\n' "exps/domainnet/coda_prompt_domainnet_official.json" ;;
        domainnet:slca) printf '%s\n' "exps/domainnet/slca_domainnet_official.json" ;;
        domainnet:ssiat) printf '%s\n' "exps/domainnet/ssiat_domainnet_official.json" ;;
        domainnet:fecam) printf '%s\n' "exps/domainnet/fecam_domainnet_official.json" ;;
        domainnet:ranpac) printf '%s\n' "exps/domainnet/ranpac_domainnet_official.json" ;;
        domainnet:aper_adapter) printf '%s\n' "exps/domainnet/aper_adapter_domainnet_official.json" ;;
        domainnet:aper_finetune) printf '%s\n' "exps/domainnet/aper_finetune_domainnet_official.json" ;;
        domainnet:aper_ssf) printf '%s\n' "exps/domainnet/aper_ssf_domainnet_official.json" ;;
        domainnet:aper_vpt_deep) printf '%s\n' "exps/domainnet/aper_vpt_deep_domainnet_official.json" ;;
        domainnet:aper_vpt_shallow) printf '%s\n' "exps/domainnet/aper_vpt_shallow_domainnet_official.json" ;;
        domainnet:ease) printf '%s\n' "exps/domainnet/ease_domainnet_official.json" ;;
        domainnet:cofima) printf '%s\n' "exps/domainnet/cofima_domainnet_official.json" ;;
        domainnet:mos) printf '%s\n' "exps/domainnet/mos_domainnet_official.json" ;;
        domainnet:tuna) printf '%s\n' "exps/domainnet/tuna_domainnet_official.json" ;;
        domainnet:min) printf '%s\n' "exps/domainnet/min_domainnet_official.json" ;;
        *)
            echo "[ERROR] No config for ${dataset}/${method}" >&2
            exit 1
            ;;
    esac
}

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

run_one() {
    local dataset="$1"
    local method="$2"
    local config_path
    config_path="$(config_for "${dataset}" "${method}")"
    run_config "${config_path}" "bash-official-hparams-${dataset}-${method}"
}

DATASET="$(normalize_dataset "${DATASET}")"
if [[ "${DATASET}" == "all" ]]; then
    DATASETS=(imagenetr omnibenchmark domainnet)
else
    DATASETS=("${DATASET}")
fi

METHODS=()
while IFS= read -r method; do
    METHODS+=("${method}")
done < <(expand_method "${METHOD}")

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_one "${dataset}" "${method}"
    done
done
