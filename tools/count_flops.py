#!/usr/bin/env python3
"""
Fast inference FLOPs counter.
Creates each model (no task simulation), measures single-pass FLOPs with fvcore,
then multiplies by the number of backbone passes determined from each method's eval_task.

Usage: python tools/count_flops.py
"""

import sys, os, warnings, logging, io
from contextlib import redirect_stdout, redirect_stderr
from collections import OrderedDict

import torch
from torch import nn

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ.update(TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_PROGRESS_BARS="1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── TIMM compatibility patches ──
def _patch():
    try:
        from timm.models._pretrained import PretrainedCfg
        if not hasattr(PretrainedCfg, '__getitem__'):
            PretrainedCfg.__getitem__ = lambda s, k: getattr(s, k)
    except: pass
    try:
        import timm.models._builder as _b
        _o = _b.build_model_with_cfg
        def _f(m, v, p=False, **kw):
            kw.pop('pretrained_custom_load', None)
            kw.pop('pretrained_strict_load', None)
            return _o(m, v, p, **kw)
        _b.build_model_with_cfg = _f
    except: pass
_patch()

from fvcore.nn import FlopCountAnalysis
from utils import factory

# ══════════════════════════════════════════════════════════════════════════
# Inference pass counts – verified from each method's eval_task source code
# ══════════════════════════════════════════════════════════════════════════
# L2P:        l2p.py:217  – 1× backbone (single prompt ViT forward)
# DualPrompt: dualprompt.py:217 – 1× backbone  
# CODA-Prompt:coda_prompt.py:162 – 1× backbone
# APER:       base.py:199 – 1× backbone
# SLCA:       base.py:199 – 1× backbone
# SSIAT:      ssiat.py:202 – 1× backbone
# FeCAM:      fecam.py:123 – 1× backbone
# RanPAC:     base.py:199 – 1× backbone
# ACIL:       base.py:199 – 1× backbone (analytical classifier)
# MiN:        base.py:199 – 1× backbone
# EASE:       ease.py:375 – 1× backbone (structured cosine fc)
# COFiMA:     base.py:199 – 1× backbone (same as SLCA)
# MOS:        mos.py:389 – 1 orig + range(_cur_task+1) = 1+10 = 11× backbone
# TUNA:       tuna.py:426 – range(_cur_task+1) + 1 merged = 10+1 = 11× backbone
# SPIE:       spie.py:292 – 1 shared + 1 batched multi-expert = 2× backbone

INFERENCE = OrderedDict({
    "SLCA":           1,
    "SSIAT":          1,
    "ACIL":           1,
    "MiN":            1,
    "RanPAC":         1,
    "APER (adapter)": 1,
    "FeCAM":          1,
    "EASE":           1,
    "COFiMA":         1,
    "SPIE":           2,
    "CODA-Prompt":    1,
    "DualPrompt":     1,
    "L2P":            1,
    "MOS":            11,
    "TUNA":           11,
})

# ══════════════════════════════════════════════════════════════════════════
# Model configs (minimal, pretrained=False for speed)
# ══════════════════════════════════════════════════════════════════════════
B = {
    "dataset": "domainnet", "shuffle": False,
    "init_cls": 20, "increment": 20,
    "device": ["cpu"], "seed": [1993],
    "batch_size": 1, "nb_classes": 200, "nb_tasks": 10,
    "memory_size": 0, "memory_per_class": 0, "fixed_memory": False,
    "pretrained": False, "num_workers": 0, "eval_workers": 0,
    "weight_decay": 0.0005, "min_lr": 0.0,
    "optimizer": "sgd", "scheduler": "cosine",
    "domainnet_root": "./data/domainnet", "domainnet_protocol": "official",
}

CONFIGS = OrderedDict([
    ("L2P", {**B,
        "model_name": "l2p", "backbone_type": "vit_base_patch16_224_l2p",
        "get_original_backbone": True, "tuned_epoch": 1, "init_lr": 0.001,
        "optimizer": "adam", "scheduler": "constant", "reinit_optimizer": True,
        "freeze": ["blocks","patch_embed","cls_token","norm","pos_embed"],
        "global_pool": "token", "head_type": "prompt",
        "prompt_pool": True, "size": 10, "length": 5, "top_k": 5,
        "prompt_key": True, "prompt_key_init": "uniform",
        "batchwise_prompt": True, "embedding_key": "cls",
        "pull_constraint": True, "pull_constraint_coeff": 0.1,
        "drop": 0, "drop_path": 0, "initializer": "uniform",
        "shared_prompt_pool": False, "shared_prompt_key": False,
        "use_prompt_mask": False, "predefined_key": "",
    }),
    ("DualPrompt", {**B,
        "model_name": "dualprompt", "backbone_type": "vit_base_patch16_224_dualprompt",
        "get_original_backbone": True, "tuned_epoch": 1, "init_lr": 0.001,
        "optimizer": "adam", "scheduler": "constant", "reinit_optimizer": True,
        "freeze": ["blocks","patch_embed","cls_token","norm","pos_embed"],
        "global_pool": "token", "head_type": "token",
        "use_g_prompt": True, "g_prompt_length": 5, "g_prompt_layer_idx": [0,1],
        "use_e_prompt": True, "e_prompt_layer_idx": [2,3,4],
        "use_prefix_tune_for_g_prompt": True, "use_prefix_tune_for_e_prompt": True,
        "prompt_pool": True, "size": 10, "length": 5, "top_k": 1,
        "prompt_key": True, "prompt_key_init": "uniform",
        "batchwise_prompt": True, "embedding_key": "cls",
        "pull_constraint": True, "pull_constraint_coeff": 0.1,
        "drop": 0, "drop_path": 0, "initializer": "uniform",
        "shared_prompt_pool": True, "shared_prompt_key": False,
        "use_prompt_mask": True, "same_key_value": False, "predefined_key": "",
    }),
    ("CODA-Prompt", {**B,
        "model_name": "coda_prompt", "backbone_type": "vit_base_patch16_224_coda_prompt",
        "tuned_epoch": 1, "init_lr": 0.001, "optimizer": "adam", "scheduler": "cosine",
        "reinit_optimizer": True, "prompt_param": [100, 8, 0], "drop": 0, "drop_path": 0,
    }),
    ("SSIAT", {**B,
        "model_name": "ssiat", "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
        "ssca": True, "ca": True,
        "init_epochs": 1, "inc_epochs": 1, "ca_epochs": 1,
        "init_lr": 0.01, "ffn_num": 64, "scale": 20, "margin": 0,
    }),
    ("RanPAC", {**B,
        "model_name": "ranpac", "backbone_type": "pretrained_vit_b16_224_in21k_vpt",
        "tuned_epoch": 1, "init_lr": 0.01, "ffn_num": 64,
        "vpt_type": "shallow", "prompt_token_num": 30,
        "use_RP": True, "M": 10000,
    }),
    ("ACIL", {**B,
        "model_name": "acil", "backbone_type": "pretrained_vit_b16_224_in21k",
        "batch_size": 128, "fit_batch_size": 128,
        "buffer_size": 2048, "gamma": 1, "use_input_norm": False,
    }),
    ("MiN", {**B,
        "model_name": "min", "backbone_type": "pretrained_vit_b16_224_in21k_min",
        "optimizer_type": "sgd", "scheduler_type": "step",
        "init_epochs": 1, "init_lr": 0.001, "init_batch_size": 128,
        "init_weight_decay": 0.0005, "epochs": 1, "lr": 0.001,
        "weight_decay": 0.0005, "batch_size": 128, "buffer_batch": 1500,
        "fit_epochs": 1, "hidden_dim": 192, "buffer_size": 16384, "gamma": 500,
    }),
    ("EASE", {**B,
        "model_name": "ease", "backbone_type": "vit_base_patch16_224_ease",
        "init_epochs": 1, "later_epochs": 1, "init_lr": 0.01, "later_lr": 0.01,
        "vpt_type": "Deep", "prompt_token_num": 5, "ffn_num": 64,
        "use_diagonal": False, "recalc_sim": True, "alpha": 1, "use_init_ptm": True,
        "beta": 1, "use_reweight": True, "moni_adam": True, "adapter_num": 1,
        "use_old_data": False,
    }),
    ("COFiMA", {**B,
        "model_name": "cofima", "backbone_type": "vit_base_patch16_224",
        "model_postfix": "50e", "lrate": 0.01, "lrate_decay": 0.1,
        "epochs": 1, "ca_epochs": 1, "ca_with_logit_norm": 0.1,
        "milestones": [40], "fisher_weighting": True, "wt_lambda": 0.4,
        "drop": 0, "drop_path": 0,
        "prefix": "cofima-test",
    }),
    ("APER (adapter)", {**B,
        "model_name": "aper_adapter", "backbone_type": "pretrained_vit_b16_224_in21k_adapter",
        "tuned_epoch": 1, "init_lr": 0.01, "ffn_num": 64,
        "vpt_type": "shallow", "prompt_token_num": 30,
    }),
    ("SLCA", {**B,
        "model_name": "slca", "backbone_type": "vit_base_patch16_224",
        "model_postfix": "50e", "lrate": 0.01, "lrate_decay": 0.1,
        "epochs": 1, "ca_epochs": 1, "ca_with_logit_norm": 0.1,
        "milestones": [40], "drop": 0, "drop_path": 0,
    }),
    ("FeCAM", {**B,
        "model_name": "fecam", "backbone_type": "pretrained_vit_b16_224_adapter",
        "tuned_epoch": 1, "init_lr": 0.01, "ffn_num": 64, "drop": 0, "drop_path": 0,
    }),
    ("MOS", {**B,
        "model_name": "mos", "backbone_type": "vit_base_patch16_224_mos",
        "tuned_epoch": 1, "init_lr": 0.01, "batch_size": 1,
        "reg": 0.1, "adapter_momentum": 0.1, "ensemble": True,
        "crct_epochs": 1, "ca_lr": 0.005,
        "ca_storage_efficient_method": "variance", "ffn_num": 16,
        "reinit_optimizer": True, "drop": 0, "drop_path": 0,
        "init_milestones": [10], "init_lr_decay": 0.1,
    }),
    ("TUNA", {**B,
        "model_name": "tuna", "backbone_type": "vit_base_patch16_224_in21k_tuna",
        "tuned_epoch": 1, "init_lr": 0.02, "reg": 0.001, "use_orth": True,
        "crct_epochs": 1, "ca_lr": 0.005,
        "ca_storage_efficient_method": "variance", "decay": True,
        "r": 16, "scale": 20, "m": 0,
        "reinit_optimizer": True, "drop": 0, "drop_path": 0,
        "init_milestones": [10], "init_lr_decay": 0.1,
        "ca_storage_efficient_method_choices": ["covariance", "variance"],
    }),
    ("SPIE", {**B,
        "model_name": "spie", "backbone_type": "vit_base_patch16_224_in21k_spie",
        "decay": True, "r": 16, "scale": 20.0, "m": 0.0,
        "expert_tokens": 4, "expert_residual_scale": 0.5,
        "shared_lora_rank": 8, "shared_lora_alpha": 1.0, "use_shared_adapter": True,
        "vera_rank": 256, "vera_dropout": 0.0, "vera_d_initial": 0.1,
        "vera_save_projection": True,
        "share_lora_weight_decay": 0.0005, "expert_head_weight_decay": 0.0005,
        "task0_shared_epochs": 1, "task0_shared_lr": 0.02,
        "task0_expert_epochs": 1, "task0_expert_lr": 0.02,
        "incremental_expert_epochs": 1, "incremental_expert_lr": 0.03,
        "shared_cls_epochs": 1, "shared_cls_lr": 0.02,
        "shared_cls_weight_decay": 0.0005, "shared_cls_crct_epochs": 1,
        "shared_cls_ca_lr": 0.005, "freeze_shared_lora_after_task0": True,
        "spie_backbone_dataparallel": False,
        "covariance_regularization": 0.0001, "max_covariance_retry_power": 6,
        "ca_storage_efficient_method": "variance",
        "expert_shape_distill_lambda": 0.1, "expert_shape_distill_temperature": 2.0,
        "expert_shape_reg_cap_ratio": 0.25,
        "posterior_task_temperature": 1.0, "posterior_expert_temperature": 1.0,
        "posterior_shared_temperature": 1.0, "posterior_alpha": 0.2,
        "posterior_router": "prototype_activation",
    }),
])

# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("  Inference FLOPs per sample (fvcore measured, DomainNet 10-step, batch=1)")
    print("  Total = measured single-pass × passes from eval_task source code")
    print("=" * 90)

    results = []
    x = torch.randn(1, 3, 224, 224)

    for name, args in CONFIGS.items():
        print(f"\n{'─'*75}\n  {name}\n{'─'*75}")
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                model = factory.get_model(args["model_name"], args)

            net = model._network
            # Methods where fc is None at init: call update_fc with total classes
            if not hasattr(net, 'fc') or net.fc is None:
                if hasattr(net, 'update_fc'):
                    net.update_fc(200)
            net.eval()

            # Measure single forward pass
            with torch.no_grad():
                gflops = FlopCountAnalysis(net, x).total() / 1e9

            passes = INFERENCE[name]
            total = gflops * passes

            print(f"  Single pass (fvcore): {gflops:>8.2f} GFLOPs")
            print(f"  Inference passes:     {passes}×")
            print(f"  TOTAL per sample:     {total:>8.2f} GFLOPs")

            results.append((name, gflops, passes, total))
            del model

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── Summary table ──
    if results:
        print(f"\n\n{'='*90}")
        print(f"  SUMMARY (T=10 tasks, batch=1, 224×224)")
        print(f"{'='*90}")
        print(f"{'Method':<20s} {'Passes':>6s} {'Single(G)':>10s} {'Total(G)':>10s}  vs Baseline(17.6G)")
        print('─' * 80)

        baseline = 17.6
        for name, single, passes, total in sorted(results, key=lambda x: x[3]):
            vs = f"{total/baseline:.1f}×"
            print(f"{name:<20s} {passes:>4}×  {single:>9.2f}  {total:>9.2f}  {vs:>8s}")

        print(f"\n  Baseline: Frozen ViT-B/16 ~17.6 GFLOPs (single pass)")
        print(f"  MOS/TUNA multi-pass count = 1 + (_cur_task + 1) = 1 + 10 = 11 for T=10")
        print(f"  SPIE uses forward_multi_expert_features → 2 passes (shared + batched experts)")
        print(f"  Prompt methods (L2P/DualPrompt/CODA) insert tokens → higher per-pass cost")


if __name__ == "__main__":
    main()
