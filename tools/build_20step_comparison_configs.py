#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path


METHODS = [
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
]

TARGETS = {
    "cifar224": {
        "source_dir": Path("exps/cifar224"),
        "prefix_dataset": "cifar100",
        "init_cls": 5,
        "increment": 5,
    },
    "cub": {
        "source_dir": Path("exps/cub"),
        "prefix_dataset": "cub",
        "init_cls": 10,
        "increment": 10,
    },
}


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


def build_config(target, method, overwrite=False):
    target_info = TARGETS[target]
    source_path = target_info["source_dir"] / f"{method}.json"
    output_path = target_info["source_dir"] / f"{method}-20step.json"

    if output_path.exists() and not overwrite:
        return output_path, source_path, "exists"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source config: {source_path}")

    cfg = load_json(source_path)
    normalized = dict(cfg)
    normalized["prefix"] = f"official-{method}-{target_info['prefix_dataset']}-20step"
    normalized["dataset"] = target
    normalized["init_cls"] = target_info["init_cls"]
    normalized["increment"] = target_info["increment"]
    normalized["model_name"] = method
    normalized["device"] = ["0"]
    normalized["swanlab"] = True
    normalized["swanlab_project"] = "SPIE"
    normalized["swanlab_mode"] = "online"

    dump_json(output_path, normalized)
    return output_path, source_path, "written"


def main():
    parser = argparse.ArgumentParser(
        description="Build CIFAR-100 and CUB-200 20-step comparison configs."
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing 20-step configs.")
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
