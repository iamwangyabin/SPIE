import copy
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "exps" / "imagenetr" / "spie.json"
OUT_DIR = ROOT / "exps" / "imagenetr" / "spie_ablation"


def load_base():
    with BASE_CONFIG.open() as handle:
        cfg = json.load(handle)

    swanlab_project = os.environ.get("SWANLAB_PROJECT", "SPIE-ablation")
    cfg.update(
        {
            "prefix": "spie-ablation-imagenetr",
            "model_name": "spie_ablation",
            "backbone_type": "vit_base_patch16_224_in21k_spie_ablation",
            "swanlab": True,
            "swanlab_project": swanlab_project,
            "swanlab_mode": "online",
            "swanlab_tags": ["spie", "ablation", "imagenetr"],
            "ca_storage_efficient_method": "diag_lowrank",
            "ca_lowrank_rank": 16,
            "spie_eval_fusion_strategy": "probability_fusion",
            "synthetic_samples_per_class": 160,
            "expert_adapter_type": "vera",
            "expert_lora_rank": 8,
            "expert_lora_alpha": 1.0,
            "spie_disable_experts": False,
            "spie_disable_replay": False,
        }
    )
    if os.environ.get("SWANLAB_WORKSPACE"):
        cfg["swanlab_workspace"] = os.environ["SWANLAB_WORKSPACE"]
    if os.environ.get("SWANLAB_LOGDIR"):
        cfg["swanlab_logdir"] = os.environ["SWANLAB_LOGDIR"]
    return cfg


def make_config(base, group, name, updates, table, purpose):
    cfg = copy.deepcopy(base)
    cfg.update(updates)
    cfg["prefix"] = f"spie-ablation-{group}-{name}"
    cfg["swanlab_group"] = f"spie_ablation__imagenetr_t10_vitb16in21k__{group}"
    cfg["swanlab_experiment_name"] = f"spie_ablation__imagenetr_t10_vitb16in21k__{group}__{name}"
    cfg["swanlab_tags"] = ["spie", "ablation", "imagenetr", group, name]
    cfg["swanlab_description"] = f"{table}: {purpose}"
    cfg["ablation_group"] = group
    cfg["ablation_name"] = name
    cfg["paper_table"] = table
    cfg["ablation_purpose"] = purpose
    return cfg


def write_config(group, name, cfg):
    group_dir = OUT_DIR / group
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / f"{name}.json"
    with path.open("w") as handle:
        json.dump(cfg, handle, indent=4)
        handle.write("\n")
    return path


def main():
    base = load_base()
    specs = []

    specs.extend(
        [
            (
                "core_component",
                "no_per_task_experts",
                {
                    "spie_disable_experts": True,
                    "task0_expert_epochs": 0,
                    "incremental_expert_epochs": 0,
                    "spie_eval_fusion_strategy": "shared_only",
                },
                "tab:component_ablation",
                "Remove per-task experts and use the calibrated shared classifier only.",
            ),
            (
                "core_component",
                "no_lowrank_distributional_replay",
                {
                    "spie_disable_replay": True,
                    "shared_cls_crct_epochs": 0,
                },
                "tab:component_ablation",
                "Disable low-rank distributional replay and prototype calibration.",
            ),
            (
                "core_component",
                "hard_routing",
                {"spie_eval_fusion_strategy": "hard_argmax"},
                "tab:component_ablation",
                "Replace soft task posterior with hard prototype-activation routing.",
            ),
            (
                "core_component",
                "full_method",
                {},
                "tab:component_ablation",
                "Full SPiE with per-task experts, low-rank replay, and probability-space fusion.",
            ),
        ]
    )

    specs.extend(
        [
            (
                "cov_format",
                "no_replay",
                {"spie_disable_replay": True, "shared_cls_crct_epochs": 0},
                "tab:cov_format",
                "No replay statistics or classifier-alignment replay.",
            ),
            (
                "cov_format",
                "mean_only",
                {"ca_storage_efficient_method": "mean"},
                "tab:cov_format",
                "Store class means only and replay deterministic prototypes.",
            ),
            (
                "cov_format",
                "diagonal",
                {"ca_storage_efficient_method": "variance"},
                "tab:cov_format",
                "Store class means plus diagonal covariance.",
            ),
            (
                "cov_format",
                "diag_lowrank_r4",
                {"ca_storage_efficient_method": "diag_lowrank", "ca_lowrank_rank": 4},
                "tab:cov_format",
                "Store diagonal covariance plus rank-4 low-rank residual.",
            ),
            (
                "cov_format",
                "diag_lowrank_r8",
                {"ca_storage_efficient_method": "diag_lowrank", "ca_lowrank_rank": 8},
                "tab:cov_format",
                "Store diagonal covariance plus rank-8 low-rank residual.",
            ),
            (
                "cov_format",
                "diag_lowrank_r16",
                {"ca_storage_efficient_method": "diag_lowrank", "ca_lowrank_rank": 16},
                "tab:cov_format",
                "Store diagonal covariance plus rank-16 low-rank residual.",
            ),
            (
                "cov_format",
                "full_covariance",
                {"ca_storage_efficient_method": "covariance"},
                "tab:cov_format",
                "Store full covariance per class.",
            ),
        ]
    )

    specs.extend(
        [
            (
                "adapter_design",
                "no_adapter",
                {"expert_adapter_type": "none"},
                "supp:adapter_design",
                "Remove expert MLP adapters and keep expert tokens/head only.",
            ),
            (
                "adapter_design",
                "lora_tuned",
                {"expert_adapter_type": "lora", "expert_lora_rank": 8, "expert_lora_alpha": 1.0},
                "supp:adapter_design",
                "Use trainable LoRA expert adapters instead of VeRA.",
            ),
            (
                "adapter_design",
                "vera_r256",
                {"expert_adapter_type": "vera", "vera_rank": 256},
                "supp:adapter_design",
                "Use the paper default VeRA expert adapters.",
            ),
        ]
    )

    for rank in [64, 128, 256, 384, 512]:
        specs.append(
            (
                "vera_rank",
                f"r{rank}",
                {"expert_adapter_type": "vera", "vera_rank": rank},
                "supp:vera_rank",
                "Measure the accuracy/parameter trade-off of the VeRA rank.",
            )
        )

    for tokens in [1, 2, 4, 8, 16]:
        specs.append(
            (
                "expert_tokens",
                f"k{tokens}",
                {"expert_tokens": tokens},
                "supp:expert_tokens",
                "Measure expert token count sensitivity and compute/storage growth.",
            )
        )

    for samples in [16, 32, 64, 128, 256]:
        specs.append(
            (
                "synthetic_samples",
                f"per_class_{samples}",
                {"synthetic_samples_per_class": samples},
                "supp:synthetic_samples",
                "Measure replay sample count sensitivity for prototype calibration.",
            )
        )

    specs.append(
        (
            "fusion_strategy",
            "all_strategies",
            {
                "spie_eval_fusion_strategy": "probability_fusion",
                "spie_eval_variants": [
                    "hard_argmax",
                    "logit_concat_softmax",
                    "logit_weighted",
                    "probability_fusion",
                    "uniform_q",
                ],
            },
            "supp:fusion_strategy",
            "Evaluate hard routing, logit fusion, probability fusion, and uniform expert averaging in one run.",
        )
    )

    for tau, tau_e in [(0.5, 1), (0.5, 2), (1, 1), (1, 2), (2, 1), (2, 2)]:
        tau_name = str(tau).replace(".", "p")
        specs.append(
            (
                "temperature",
                f"tau{tau_name}_taue{tau_e}",
                {
                    "posterior_task_temperature": tau,
                    "posterior_expert_temperature": tau_e,
                },
                "supp:temperature",
                "Measure sensitivity to task posterior and expert probability temperatures.",
            )
        )

    manifest = []
    for group, name, updates, table, purpose in specs:
        cfg = make_config(base, group, name, updates, table, purpose)
        path = write_config(group, name, cfg)
        manifest.append(path.relative_to(ROOT).as_posix())

    manifest_path = OUT_DIR / "manifest.txt"
    with manifest_path.open("w") as handle:
        handle.write("\n".join(manifest))
        handle.write("\n")

    print(f"Wrote {len(manifest)} SPIE ablation config(s) under {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
