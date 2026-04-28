#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path


METHODS = [
    "spie",
    "l2p",
    "acil",
    "fecam",
    "mos",
    "min",
    "aper",
    "ssiat",
    "tuna",
    "dualprompt",
    "slca",
    "coda_prompt",
    "ranpac",
    "ease",
    "cofima",
]

TARGETS = {
    "imagenetr": {
        "config_dir": Path("exps/imagenetr"),
        "output_dir": Path("exps/imagenetr/official"),
        "prefix_dataset": "imagenetr",
        "init_cls": 20,
        "increment": 20,
    },
    "cifar224": {
        "config_dir": Path("exps/cifar224"),
        "output_dir": Path("exps/cifar224/official"),
        "prefix_dataset": "cifar100",
        "init_cls": 10,
        "increment": 10,
    },
    "cub": {
        "config_dir": Path("exps/cub"),
        "output_dir": Path("exps/cub/official"),
        "prefix_dataset": "cub",
        "init_cls": 20,
        "increment": 20,
    },
}


FALLBACK_SOURCES = [
    Path("exps/imagenetr/generated"),
    Path("exps/omnibenchmark/official"),
    Path("exps/domainnet/official"),
]


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
    tmp_path.replace(path)


def iter_direct_configs(config_dir):
    for path in sorted(config_dir.glob("*.json")):
        try:
            yield path, load_json(path)
        except json.JSONDecodeError:
            continue


def find_same_dataset_source(target, method):
    config_dir = TARGETS[target]["config_dir"]
    init_cls = TARGETS[target]["init_cls"]
    increment = TARGETS[target]["increment"]

    method_configs = [
        (path, cfg)
        for path, cfg in iter_direct_configs(config_dir)
        if cfg.get("model_name") == method and cfg.get("dataset") == target
    ]
    for path, cfg in method_configs:
        if cfg.get("init_cls") == init_cls and cfg.get("increment") == increment:
            return path, cfg
    if method_configs:
        return method_configs[0]
    return None, None


def find_fallback_source(method):
    candidates = [
        f"{method}_imagenetr.json",
        f"{method}.json",
    ]
    for source_dir in FALLBACK_SOURCES:
        for name in candidates:
            path = source_dir / name
            if path.exists():
                cfg = load_json(path)
                if cfg.get("model_name") == method:
                    return path, cfg
    return None, None


def normalize_config(cfg, target, method):
    target_info = TARGETS[target]
    normalized = dict(cfg)

    normalized["prefix"] = f"official-{method}-{target_info['prefix_dataset']}-10step"
    normalized["dataset"] = target
    normalized["memory_size"] = 0
    normalized["memory_per_class"] = 0
    normalized["fixed_memory"] = False
    normalized["shuffle"] = True
    normalized["init_cls"] = target_info["init_cls"]
    normalized["increment"] = target_info["increment"]
    normalized["model_name"] = method
    normalized["device"] = ["0"]
    normalized["seed"] = [1993, 1996, 1997] if method == "ssiat" else [1993]
    normalized["swanlab"] = True
    normalized["swanlab_project"] = "SPIE"
    normalized["swanlab_mode"] = "online"

    normalized.pop("domainnet_protocol", None)
    normalized.pop("domainnet_root", None)
    normalized.pop("omnibenchmark_root", None)

    if "backbone_type" not in normalized and "convnet_type" in normalized:
        normalized["backbone_type"] = normalized["convnet_type"]

    if method == "ssiat":
        tuned_epoch = normalized.get("tuned_epoch")
        normalized.setdefault("init_epochs", tuned_epoch if tuned_epoch is not None else 20)
        normalized.setdefault("inc_epochs", tuned_epoch if tuned_epoch is not None else 10)
        normalized.setdefault("ca_epochs", 5)

    return normalized


def build_config(target, method, overwrite=False):
    output_path = TARGETS[target]["output_dir"] / f"{method}.json"
    if output_path.exists() and not overwrite:
        return output_path, None, "exists"

    source_path, source_cfg = find_same_dataset_source(target, method)
    if source_cfg is None:
        source_path, source_cfg = find_fallback_source(method)
    if source_cfg is None:
        raise FileNotFoundError(f"No source config found for {method} on {target}")

    normalized = normalize_config(source_cfg, target, method)
    dump_json(output_path, normalized)
    return output_path, source_path, "written"


def main():
    parser = argparse.ArgumentParser(description="Build 10-step comparison configs for ImageNet-R, CIFAR100, and CUB-200.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite configs even when output files already exist.")
    args = parser.parse_args()

    for target in TARGETS:
        for method in METHODS:
            output_path, source_path, status = build_config(target, method, overwrite=args.overwrite)
            if status == "exists":
                print(f"exists  {output_path}")
            else:
                print(f"wrote   {output_path}  <-  {source_path}")


if __name__ == "__main__":
    main()
