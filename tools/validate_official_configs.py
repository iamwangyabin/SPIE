#!/usr/bin/env python
import argparse
import json
from pathlib import Path


DATASETS = {
    "omnibenchmark": (30, 30),
    "domainnet": (20, 20),
    "imagenetr": (20, 20),
    "cifar224": (10, 10),
    "cub": (20, 20),
}

METHODS = {
    "l2p",
    "dualprompt",
    "coda_prompt",
    "acil",
    "slca",
    "ssiat",
    "fecam",
    "ranpac",
    "aper",
    "ease",
    "cofima",
    "mos",
    "tuna",
    "min",
    "spie",
}

COMMON_REQUIRED = {
    "prefix",
    "dataset",
    "memory_size",
    "memory_per_class",
    "fixed_memory",
    "shuffle",
    "init_cls",
    "increment",
    "model_name",
    "backbone_type",
    "device",
    "seed",
}

METHOD_REQUIRED = {
    "l2p": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "reinit_optimizer",
        "get_original_backbone",
        "freeze",
        "pretrained",
        "drop",
        "drop_path",
        "prompt_pool",
        "size",
        "length",
        "top_k",
        "prompt_key",
        "prompt_key_init",
        "use_prompt_mask",
        "batchwise_prompt",
        "embedding_key",
        "pull_constraint",
        "pull_constraint_coeff",
    },
    "dualprompt": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "reinit_optimizer",
        "get_original_backbone",
        "freeze",
        "pretrained",
        "drop",
        "drop_path",
        "use_g_prompt",
        "g_prompt_length",
        "g_prompt_layer_idx",
        "use_prefix_tune_for_g_prompt",
        "use_e_prompt",
        "e_prompt_layer_idx",
        "use_prefix_tune_for_e_prompt",
        "prompt_pool",
        "size",
        "length",
        "top_k",
        "prompt_key",
        "prompt_key_init",
        "use_prompt_mask",
        "batchwise_prompt",
        "embedding_key",
        "pull_constraint",
        "pull_constraint_coeff",
        "same_key_value",
    },
    "coda_prompt": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "pretrained",
        "drop",
        "drop_path",
        "prompt_param",
    },
    "acil": {"batch_size", "fit_batch_size", "buffer_size", "gamma", "pretrained", "num_workers"},
    "slca": {
        "lrate",
        "weight_decay",
        "lrate_decay",
        "batch_size",
        "epochs",
        "ca_epochs",
        "ca_with_logit_norm",
        "milestones",
    },
    "ssiat": {
        "ssca",
        "ca",
        "ssiat_official_seed_behavior",
        "init_epochs",
        "inc_epochs",
        "ca_epochs",
        "init_lr",
        "batch_size",
        "num_workers",
        "eval_workers",
        "weight_decay",
        "min_lr",
        "ffn_num",
        "optimizer",
        "scale",
        "margin",
    },
    "fecam": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "ffn_num",
        "optimizer",
        "vpt_type",
        "prompt_token_num",
    },
    "ranpac": {
        "use_simplecil",
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "ffn_num",
        "optimizer",
        "vpt_type",
        "prompt_token_num",
        "use_RP",
        "M",
    },
    "aper": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "ffn_num",
        "optimizer",
    },
    "ease": {
        "init_epochs",
        "init_lr",
        "later_epochs",
        "later_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "pretrained",
        "vpt_type",
        "prompt_token_num",
        "ffn_num",
        "use_diagonal",
        "recalc_sim",
        "alpha",
        "use_init_ptm",
        "beta",
        "use_old_data",
        "use_reweight",
        "moni_adam",
        "adapter_num",
    },
    "cofima": {
        "lrate",
        "weight_decay",
        "lrate_decay",
        "batch_size",
        "epochs",
        "ca_epochs",
        "ca_with_logit_norm",
        "milestones",
        "fisher_weighting",
        "wt_lambda",
    },
    "mos": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "reinit_optimizer",
        "init_milestones",
        "init_lr_decay",
        "reg",
        "adapter_momentum",
        "ensemble",
        "crct_epochs",
        "ca_lr",
        "ca_storage_efficient_method",
        "n_centroids",
        "pretrained",
        "drop",
        "drop_path",
        "ffn_num",
    },
    "tuna": {
        "tuned_epoch",
        "init_lr",
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "reinit_optimizer",
        "init_milestones",
        "init_lr_decay",
        "reg",
        "use_orth",
        "crct_epochs",
        "ca_lr",
        "ca_storage_efficient_method",
        "decay",
        "pretrained",
        "drop",
        "drop_path",
        "r",
        "scale",
        "m",
    },
    "min": {
        "pretrained",
        "optimizer_type",
        "scheduler_type",
        "init_epochs",
        "init_lr",
        "init_batch_size",
        "init_weight_decay",
        "epochs",
        "lr",
        "batch_size",
        "buffer_batch",
        "fit_epochs",
        "weight_decay",
        "num_workers",
        "hidden_dim",
        "buffer_size",
        "gamma",
    },
    "spie": {
        "batch_size",
        "weight_decay",
        "min_lr",
        "optimizer",
        "scheduler",
        "ca_storage_efficient_method",
        "decay",
        "r",
        "scale",
        "m",
        "expert_tokens",
        "expert_residual_scale",
        "shared_lora_rank",
        "shared_lora_alpha",
        "use_shared_adapter",
        "share_lora_weight_decay",
        "expert_head_weight_decay",
        "task0_shared_epochs",
        "task0_shared_lr",
        "task0_expert_epochs",
        "task0_expert_lr",
        "incremental_expert_epochs",
        "incremental_expert_lr",
        "shared_cls_epochs",
        "shared_cls_lr",
        "shared_cls_weight_decay",
        "shared_cls_crct_epochs",
        "shared_cls_ca_lr",
        "freeze_shared_lora_after_task0",
        "spie_backbone_dataparallel",
        "covariance_regularization",
        "max_covariance_retry_power",
        "posterior_task_temperature",
        "posterior_expert_temperature",
        "posterior_shared_temperature",
        "posterior_alpha",
        "posterior_router",
    },
}


def load_config(path):
    with path.open() as handle:
        return json.load(handle)


def default_config_paths():
    paths = []
    for dataset in DATASETS:
        paths.extend((Path("exps") / dataset).glob("*.json"))
    return sorted(paths)


def validate(path):
    cfg = load_config(path)
    errors = []

    missing_common = sorted(COMMON_REQUIRED - cfg.keys())
    if missing_common:
        errors.append(f"missing common keys: {missing_common}")

    dataset = cfg.get("dataset")
    if dataset not in DATASETS:
        errors.append(f"unsupported dataset: {dataset}")
    else:
        init_cls, increment = DATASETS[dataset]
        if cfg.get("init_cls") != init_cls or cfg.get("increment") != increment:
            errors.append(
                f"{dataset} must use init_cls={init_cls}, increment={increment}; "
                f"got {cfg.get('init_cls')}/{cfg.get('increment')}"
            )
        if dataset == "domainnet" and cfg.get("domainnet_protocol") != "official":
            errors.append("domainnet config must set domainnet_protocol='official'")
        if dataset == "omnibenchmark" and "omnibenchmark_root" not in cfg:
            errors.append("omnibenchmark config must set omnibenchmark_root")

    method = cfg.get("model_name")
    if method not in METHODS:
        errors.append(f"unsupported comparison method: {method}")
    else:
        missing_method = sorted(METHOD_REQUIRED[method] - cfg.keys())
        if missing_method:
            errors.append(f"missing {method} keys: {missing_method}")

    device = cfg.get("device")
    if device != ["0"]:
        errors.append(f"comparison configs should use device ['0']; scripts select physical GPU, got {device}")

    if method == "aper":
        backbone = str(cfg.get("backbone_type", "")).lower()
        if not any(token in backbone for token in ("adapter", "ssf", "vpt", "vit_b16_224")):
            errors.append(f"aper backbone is not routable: {cfg.get('backbone_type')}")
    if method == "acil" and int(cfg.get("buffer_size", 0)) <= 0:
        errors.append("acil buffer_size must be positive")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate comparison experiment configs before long queue runs.")
    parser.add_argument(
        "configs",
        nargs="*",
        help="Config files to validate. Defaults to exps/<dataset>/*.json for the comparison datasets.",
    )
    parser.add_argument("--require-complete", action="store_true", help="Require all method/dataset combinations.")
    args = parser.parse_args()

    if args.configs:
        paths = [Path(item) for item in args.configs]
    else:
        paths = default_config_paths()

    if not paths:
        raise SystemExit("No configs found.")

    failures = []
    seen = set()
    for path in paths:
        if not path.exists():
            failures.append(f"{path}: file does not exist")
            continue
        try:
            cfg = load_config(path)
            seen.add((cfg.get("model_name"), cfg.get("dataset")))
            errors = validate(path)
        except Exception as exc:
            errors = [f"failed to parse or validate: {exc}"]
        for error in errors:
            failures.append(f"{path}: {error}")

    if args.require_complete:
        expected = {(method, dataset) for method in METHODS for dataset in DATASETS}
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            failures.append(f"missing method/dataset combos: {missing}")
        if extra:
            failures.append(f"unexpected method/dataset combos: {extra}")

    if failures:
        raise SystemExit("\n".join(failures))

    print(f"Validated {len(paths)} comparison config(s).")


if __name__ == "__main__":
    main()
