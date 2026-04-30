#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
NOTE="${NOTE:-spie-ablation-4gpu}"
CONFIG_ROOT="${ROOT_DIR}/exps/imagenetr/spie_ablation"
LOG_ROOT="${ROOT_DIR}/logs/spie_ablation_4gpu"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-SPIE-ablation}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/generate_spie_ablation_configs.py
mkdir -p "${LOG_ROOT}"

GPU0_CONFIGS=(
    "exps/imagenetr/spie_ablation/core_component/full_method.json"
    "exps/imagenetr/spie_ablation/core_component/hard_routing.json"
    "exps/imagenetr/spie_ablation/core_component/no_lowrank_distributional_replay.json"
    "exps/imagenetr/spie_ablation/core_component/no_per_task_experts.json"
    "exps/imagenetr/spie_ablation/fusion_strategy/all_strategies.json"
    "exps/imagenetr/spie_ablation/adapter_design/lora_tuned.json"
    "exps/imagenetr/spie_ablation/adapter_design/no_adapter.json"
    "exps/imagenetr/spie_ablation/adapter_design/vera_r256.json"
)

GPU1_CONFIGS=(
    "exps/imagenetr/spie_ablation/cov_format/no_replay.json"
    "exps/imagenetr/spie_ablation/cov_format/mean_only.json"
    "exps/imagenetr/spie_ablation/cov_format/diagonal.json"
    "exps/imagenetr/spie_ablation/cov_format/diag_lowrank_r4.json"
    "exps/imagenetr/spie_ablation/cov_format/diag_lowrank_r8.json"
    "exps/imagenetr/spie_ablation/cov_format/diag_lowrank_r16.json"
    "exps/imagenetr/spie_ablation/cov_format/full_covariance.json"
)

GPU2_CONFIGS=(
    "exps/imagenetr/spie_ablation/vera_rank/r64.json"
    "exps/imagenetr/spie_ablation/vera_rank/r128.json"
    "exps/imagenetr/spie_ablation/vera_rank/r256.json"
    "exps/imagenetr/spie_ablation/vera_rank/r384.json"
    "exps/imagenetr/spie_ablation/vera_rank/r512.json"
    "exps/imagenetr/spie_ablation/temperature/tau0p5_taue1.json"
    "exps/imagenetr/spie_ablation/temperature/tau0p5_taue2.json"
    "exps/imagenetr/spie_ablation/temperature/tau1_taue1.json"
    "exps/imagenetr/spie_ablation/temperature/tau1_taue2.json"
    "exps/imagenetr/spie_ablation/temperature/tau2_taue1.json"
    "exps/imagenetr/spie_ablation/temperature/tau2_taue2.json"
)

GPU3_CONFIGS=(
    "exps/imagenetr/spie_ablation/expert_tokens/k1.json"
    "exps/imagenetr/spie_ablation/expert_tokens/k2.json"
    "exps/imagenetr/spie_ablation/expert_tokens/k4.json"
    "exps/imagenetr/spie_ablation/expert_tokens/k8.json"
    "exps/imagenetr/spie_ablation/expert_tokens/k16.json"
    "exps/imagenetr/spie_ablation/synthetic_samples/per_class_16.json"
    "exps/imagenetr/spie_ablation/synthetic_samples/per_class_32.json"
    "exps/imagenetr/spie_ablation/synthetic_samples/per_class_64.json"
    "exps/imagenetr/spie_ablation/synthetic_samples/per_class_128.json"
    "exps/imagenetr/spie_ablation/synthetic_samples/per_class_256.json"
)

ALL_CONFIGS=(
    "${GPU0_CONFIGS[@]}"
    "${GPU1_CONFIGS[@]}"
    "${GPU2_CONFIGS[@]}"
    "${GPU3_CONFIGS[@]}"
)

"${PYTHON_BIN}" tools/validate_official_configs.py "${ALL_CONFIGS[@]}"
"${PYTHON_BIN}" -c 'import json, pathlib, sys
bad = []
for item in sys.argv[1:]:
    cfg = json.loads(pathlib.Path(item).read_text())
    if cfg.get("swanlab") is not True or cfg.get("swanlab_mode") != "online":
        bad.append(item)
if bad:
    raise SystemExit("SwanLab online logging is not enabled for: " + ", ".join(bad))
print(f"SwanLab online logging enabled for {len(sys.argv) - 1} config(s).")
' "${ALL_CONFIGS[@]}"

run_queue() {
    local gpu_id="$1"
    shift
    local log_file="${LOG_ROOT}/gpu${gpu_id}.log"
    {
        echo "[GPU ${gpu_id}] starting ${#} config(s) at $(date)"
        for config in "$@"; do
            echo "[GPU ${gpu_id}] ${config} at $(date)"
            CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" main.py --config "${ROOT_DIR}/${config}" --note "${NOTE}"
        done
        echo "[GPU ${gpu_id}] finished at $(date)"
    } 2>&1 | tee "${log_file}"
}

run_queue 0 "${GPU0_CONFIGS[@]}" &
pid0=$!
run_queue 1 "${GPU1_CONFIGS[@]}" &
pid1=$!
run_queue 2 "${GPU2_CONFIGS[@]}" &
pid2=$!
run_queue 3 "${GPU3_CONFIGS[@]}" &
pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"
echo "All 4 GPU queues finished."
