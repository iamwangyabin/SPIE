#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT_DIR}/scripts/train_domainnet_added.sh"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
QUEUE_LOG_DIR="${ROOT_DIR}/logs/domainnet_added_queue/${TIMESTAMP}"
STATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/domainnet-added-queue.XXXXXX")"
FAIL_FILE="${STATE_DIR}/failed"

read -r -a GPUS <<< "${GPU_LIST:-0 1 2 3}"
METHODS=(
    l2p
    dualprompt
    coda_prompt
    ease
    slca
    ranpac
    fecam
    aper_adapter
    cofima
    mos
)

cleanup() {
    rm -rf "${STATE_DIR}"
}
trap cleanup EXIT

if [[ ! -f "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    exit 1
fi

if (( ${#GPUS[@]} == 0 )); then
    echo "[ERROR] No GPUs configured." >&2
    exit 1
fi

mkdir -p "${QUEUE_LOG_DIR}"
echo 0 > "${STATE_DIR}/next_index"

lock_acquire() {
    while ! mkdir "${STATE_DIR}/lock" 2>/dev/null; do
        sleep 1
    done
}

lock_release() {
    rmdir "${STATE_DIR}/lock"
}

next_method() {
    local idx

    if [[ -f "${FAIL_FILE}" ]]; then
        return 1
    fi

    lock_acquire
    idx="$(<"${STATE_DIR}/next_index")"
    if (( idx >= ${#METHODS[@]} )); then
        lock_release
        return 1
    fi
    echo $((idx + 1)) > "${STATE_DIR}/next_index"
    lock_release

    REPLY="${METHODS[$idx]}"
    return 0
}

worker() {
    local gpu="$1"
    local method log_file

    while next_method; do
        method="${REPLY}"
        log_file="${QUEUE_LOG_DIR}/gpu${gpu}_${method}.log"

        echo "[GPU ${gpu}] START ${method} -> ${log_file}"
        if CUDA_VISIBLE_DEVICES="${gpu}" bash "${RUNNER}" "${method}" > "${log_file}" 2>&1; then
            echo "[GPU ${gpu}] DONE  ${method}"
        else
            echo "[GPU ${gpu}] FAIL  ${method} -> ${log_file}" >&2
            : > "${FAIL_FILE}"
            return 1
        fi
    done
}

echo "[INFO] GPUs: ${GPUS[*]}"
echo "[INFO] Methods: ${METHODS[*]}"
echo "[INFO] Queue logs: ${QUEUE_LOG_DIR}"

pids=()
worker_count=0
for gpu in "${GPUS[@]}"; do
    if (( worker_count >= ${#METHODS[@]} )); then
        break
    fi
    worker "${gpu}" &
    pids+=("$!")
    worker_count=$((worker_count + 1))
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if (( status != 0 )) || [[ -f "${FAIL_FILE}" ]]; then
    echo "[ERROR] Queue finished with failures. Check logs under ${QUEUE_LOG_DIR}" >&2
    exit 1
fi

echo "[OK] All DomainNet added-method runs finished. Logs: ${QUEUE_LOG_DIR}"
