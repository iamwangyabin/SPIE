# SPIE ImageNet-R Ablation Suite

Default setting for all configs:

- Dataset: ImageNet-R
- Task protocol: T=10, init_cls=20, increment=20
- Backbone: ViT-B/16-IN21K
- Main method default: SPIE with per-task experts, diag+lowrank replay rank 16, probability-space fusion

Run all ablations:

```bash
GPU_ID=0 PYTHON_BIN=python scripts/train_spie_ablation_imagenetr.sh all
```

Run one group:

```bash
GPU_ID=0 PYTHON_BIN=python scripts/train_spie_ablation_imagenetr.sh core_component
GPU_ID=1 PYTHON_BIN=python scripts/train_spie_ablation_imagenetr.sh cov_format
GPU_ID=2 PYTHON_BIN=python scripts/train_spie_ablation_imagenetr.sh adapter_design
```

Available groups:

- `core_component`: main paper component ablation for `tab:component_ablation`
- `cov_format`: main paper covariance storage ablation for `tab:cov_format`
- `adapter_design`: supplementary VeRA/LoRA/no-adapter comparison
- `vera_rank`: supplementary VeRA rank sensitivity
- `expert_tokens`: supplementary expert token count sensitivity
- `synthetic_samples`: supplementary replay sample count sensitivity
- `fusion_strategy`: supplementary routing/fusion strategy comparison; one run logs all strategy variants
- `temperature`: supplementary posterior/expert temperature sensitivity

The config generator is `tools/generate_spie_ablation_configs.py`. Re-run it after changing the base ImageNet-R SPIE config.
