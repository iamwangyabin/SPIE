#!/usr/bin/env python3
"""
Adaptive training sweep for SPiE on ImageNet-A.

This is a thin wrapper over the generic SPiE training sweep so the
dataset-specific defaults match exps/spie_imga.json out of the box.
"""

from __future__ import annotations

from spie_inr_train_sweep import build_parser, run_from_args


def parse_args():
    parser = build_parser(
        description="Adaptive SPiE ImageNet-A training sweep",
        default_base_config="exps/spie_imga.json",
        default_output_dir="sweep_spie_imga_train",
        default_prefix="spie-autosweep-imga",
        default_target_dataset="imageneta",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
