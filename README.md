# ViT + LoRA Incremental Tuning Workspace

This repository is no longer maintained as the original multi-method PILOT toolbox.
It has been simplified into a small experimental codebase for a single pre-trained
Vision Transformer tuning workflow, and it will be used as the starting point for
implementing new methods.

At the current stage, the project keeps one working path based on:

- ViT backbone
- LoRA / adapter-style parameter-efficient tuning
- class-incremental training pipeline

The goal of this version is practical experimentation rather than toolbox completeness.
Most unrelated methods and configs have been removed on purpose so the codebase is easier
to modify for upcoming work.

## Current Scope

The retained code path is centered on the `tuna` method and its ViT tuning backbone.

Main components:

- `main.py`: experiment entry
- `trainer.py`: training loop across incremental tasks
- `models/tuna.py`: learner logic
- `utils/inc_net.py`: network wrapper and backbone construction
- `backbone/vit_tuna.py`: ViT tuning backbone
- `exps/tuna_cifar.json`: CIFAR-100 style experiment config
- `exps/tuna_inr.json`: ImageNet-R style experiment config

This repository should now be understood as a clean base for:

- reproducing a ViT + LoRA-style incremental tuning baseline
- inserting a new method with minimal interference from old implementations

## Environment

Recommended dependencies:

- `python>=3.9`
- `torch`
- `torchvision`
- `timm`
- `numpy`
- `scipy`
- `tqdm`
- `easydict`

Install them with your preferred environment manager.

## Quick Start

Run the default experiment:

```bash
python main.py
```

Or specify a config explicitly:

```bash
python main.py --config ./exps/tuna_cifar.json
python main.py --config ./exps/tuna_inr.json
```

## Configuration

The project currently expects JSON config files under `exps/`.

Important fields:

- `model_name`: should currently be `tuna`
- `backbone_type`: ViT backbone variant used by the tuner
- `dataset`: dataset name handled by `utils/data.py`
- `init_cls`: number of classes in the first session
- `increment`: number of classes introduced per later session
- `tuned_epoch`: tuning epochs per task
- `init_lr`: learning rate
- `batch_size`: batch size
- `r`: low-rank tuning dimension

## Dataset Notes

Supported dataset paths are defined in `utils/data.py`.

- `cifar224` uses torchvision download automatically
- `imagenetr` expects local folders under `./data/imagenet-r/`

If you want to train on another dataset, edit the corresponding path in
`utils/data.py`.

## Suggested Next Step

If you are about to implement your own method, a practical path is:

1. Copy `models/tuna.py` into a new learner file.
2. Add a new branch in `utils/factory.py`.
3. Reuse `utils/inc_net.py` and `backbone/vit_tuna.py` first, then only change the parts your method actually needs.
4. Add one new config under `exps/`.

This keeps the current baseline runnable while you iterate on your own idea.

## Status

This repo is now a research workspace, not a polished general-purpose framework.
The code is intentionally kept lightweight and only partially cleaned, so future
method development can move faster.
