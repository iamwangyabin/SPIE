#!/usr/bin/env python3
"""
Count ALL additional parameters/storage for each method beyond the frozen ViT-B/16 backbone.
Uses lightweight dummy backbone to avoid timm download/compatibility issues.
Only counts what matters: adapters, prompts, classifiers, stored statistics.

Usage: python tools/count_all_params.py
"""

import sys
import os
import warnings
import logging
import copy
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ── suppress noisy logging ───────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ══════════════════════════════════════════════════════════════════════════
# Constants (ViT-B/16)
# ══════════════════════════════════════════════════════════════════════════
D = 768          # feature dim
H = 3072         # MLP hidden dim (4*D)
L = 12           # number of transformer layers
NUM_HEADS = 12
D_HEAD = 64
C = 200          # total classes (DomainNet)
T = 10           # num tasks (init=20, inc=20)
INC = 20         # classes per task

# ViT-B backbone parameter count (timm vit_base_patch16_224)
VIT_B_PARAMS = 85_795_584  # ~85.8M (without head)


# ══════════════════════════════════════════════════════════════════════════
# Counting helpers
# ══════════════════════════════════════════════════════════════════════════

def _fmt(n):
    """Human-readable number."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _pct(n):
    return f"{n / VIT_B_PARAMS * 100:.1f}%"

# ══════════════════════════════════════════════════════════════════════════
# Per-method parameter counting (analytical, based on actual code dimensions)
# ══════════════════════════════════════════════════════════════════════════

def count_l2p():
    """L2P: Prompt pool (size=10, length=5) + key + classifier head."""
    pool_size, prompt_len = 10, 5
    # Prompt tokens (nn.Parameter): pool_size × length × D
    prompt = pool_size * prompt_len * D  # 38,400
    # Prompt keys: pool_size × D
    key = pool_size * D  # 7,680
    # Head: linear D×C + bias
    head = D * C + C  # 153,800
    learnable = prompt + key + head
    # Prompt pool is stored within backbone, head is in _network
    return {
        "learnable": learnable,
        "components": {
            "prompt_pool (10×5×768)": prompt,
            "prompt_keys (10×768)": key,
            "classifier_head (768×200)": head,
        },
    }


def count_dualprompt():
    """DualPrompt: G-prompt (2 layers) + E-prompt pool (3 layers, pool=10, len=5) + keys + head."""
    g_layers, g_len = 2, 5
    e_layers, pool_size, e_len = 3, 10, 5
    # G-prompt: per layer, prefix tuning: 2 × g_len × D (key+value pairs)
    g_prompt = g_layers * 2 * g_len * D  # 2×2×5×768 = 15,360
    # E-prompt pool: e_layers × pool_size × 2 × e_len × D
    e_pool = e_layers * pool_size * 2 * e_len * D  # 3×10×2×5×768 = 230,400
    # E-prompt key: pool_size × D
    e_key = pool_size * D  # 7,680
    # Head
    head = D * C + C  # 153,800
    learnable = g_prompt + e_pool + e_key + head
    return {
        "learnable": learnable,
        "components": {
            "G-prompt (2 layers × 5)": g_prompt,
            "E-prompt pool (3 layers × 10 × 5)": e_pool,
            "E-prompt keys (10×768)": e_key,
            "classifier_head (768×200)": head,
        },
    }


def count_coda_prompt():
    """CODA-Prompt: 5 layers × (prompt components + key + attention weights) + fc."""
    n_layers, pool_size, p_len = 5, 100, 8
    # Per layer: e_p(pool×len×D) + e_k(pool×D) + e_a(pool×D)
    per_layer = pool_size * p_len * D + 2 * pool_size * D
    # 100×8×768 + 200×768 = 614,400 + 153,600 = 768,000
    total_prompt = n_layers * per_layer
    # fc head
    fc = D * C + C
    learnable = total_prompt + fc
    return {
        "learnable": learnable,
        "components": {
            "CodaPrompt (5 layers × 100 × 8)": total_prompt,
            "fc (768×200)": fc,
        },
    }


def count_aper_adapter():
    """APER (adapter): adapter in 12 layers (ffn=64) + cosine classifier."""
    ffn = 64
    # Per layer: down(D×ffn+ffn) + up(ffn×D+D) = D*ffn+ffn + ffn*D+D = 2*D*ffn + ffn + D
    per_layer = 2 * D * ffn + ffn + D  # 2*768*64 + 64 + 768 = 98,304 + 832 = 99,136
    adapter = L * per_layer
    # CosineLinear: D×C + 1
    fc = D * C + 1
    learnable = adapter + fc
    return {
        "learnable": learnable,
        "components": {
            f"adapter (12 layers, ffn={ffn})": adapter,
            "CosineLinear (768×200)": fc,
        },
    }


def count_aper_ssf():
    """APER (SSF): scale+shift per block component (attention qkv/proj, MLP fc1/fc2, norms)."""
    # Per block: attn.qkv(3D scale+shift) + attn.proj(D scale+shift) + mlp.fc1(H scale+shift) + mlp.fc2(D scale+shift) + norm1(2D) + norm2(2D)
    # = 3D*2 + D*2 + H*2 + D*2 + D*2 + D*2 = 2*(3D + D + H + D + D + D)
    per_block = 2 * (3 * D + D + H + D + D + D)  # 2*(2304+768+3072+768+768+768) = 2*8448 = 16,896
    backbone_ssf = L * per_block
    # Patch embed SSF: 2*D
    # Head SSF: (for feature extraction only, same as norm)
    total_ssf = backbone_ssf + 2 * D
    # CosineLinear
    fc = D * C + 1
    learnable = total_ssf + fc
    return {
        "learnable": learnable,
        "components": {
            f"SSF (12 blocks, scale+shift)": total_ssf,
            "CosineLinear (768×200)": fc,
        },
    }


def count_aper_vpt():
    """APER (VPT Deep): prompt tokens at each layer (10 tokens per layer)."""
    n_tokens = 10
    prompt = L * n_tokens * D  # 12 × 10 × 768 = 92,160
    fc = D * C + 1
    learnable = prompt + fc
    return {
        "learnable": learnable,
        "components": {
            f"VPT Deep ({L} layers × {n_tokens} tokens)": prompt,
            "CosineLinear (768×200)": fc,
        },
    }


def count_acil():
    """ACIL: RandomFeatureBuffer (D×B) + weight (B×C) + R (B×B) – all as registered buffers (0 learnable)."""
    B = 2048
    rfb = D * B              # 768 × 2048 = 1,572,864
    weight = B * C           # 2048 × 200 = 409,600
    R_mat = B * B            # 2048² = 4,194,304
    stored = rfb + weight + R_mat
    return {
        "learnable": 0,
        "stored_statistics": stored,
        "components": {
            "RandomFeatureBuffer (768×2048)": rfb,
            "Weight (2048×200)": weight,
            "R precision (2048×2048)": R_mat,
        },
    }


def count_min():
    """MiN: RandomBuffer (D×B) + weight (B×C) + R (B²) + task_prototypes (T×D) + noise params."""
    B = 16384
    rfb = D * B              # 768 × 16384 = 12,582,912
    weight = B * C           # 16384 × 200 = 3,276,800
    R_mat = B * B            # 16384² = 268,435,456
    task_protos = T * D      # 10 × 768 = 7,680
    stored = rfb + weight + R_mat + task_protos
    # Noise params in vit_min: per block: norm1(2*D) + norm2(2*D) + extra noise scale/shift ≈ 4D per block
    noise_params = L * 4 * D  # ~36,864
    return {
        "learnable": noise_params,
        "stored_statistics": stored,
        "components": {
            "RandomBuffer (768×16384)": rfb,
            "Weight (16384×200)": weight,
            "R precision (16384×16384)": R_mat,
            f"Task prototypes ({T}×768)": task_protos,
            f"Noise params (12 blocks)": noise_params,
        },
    }


def count_ranpac():
    """RanPAC: adapter(ffn=64)×12 + RP matrix (D×M) + Q (M×C) + G (M²) + fc (C×M)."""
    M = 10000
    ffn = 64
    # Adapter
    per_layer = 2 * D * ffn + ffn + D
    adapter = L * per_layer  # 1,189,632
    # RP matrix: D × M
    RP = D * M               # 7,680,000
    # fc after RP: C × M + 1
    fc = C * M + 1           # 2,000,001
    # Q accumulator: M × C
    Q = M * C                # 2,000,000
    # G accumulator: M × M
    G = M * M                # 100,000,000
    learnable = adapter + fc
    stored = RP + Q + G
    return {
        "learnable": learnable,
        "stored_statistics": stored,
        "components": {
            f"Adapter (12 layers, ffn=64)": adapter,
            "RP matrix W_rand (768×10000)": RP,
            "fc after RP (200×10000)": fc,
            "Q accumulator (10000×200)": Q,
            "G accumulator (10000×10000)": G,
        },
    }


def count_slca():
    """SLCA: per-task heads (T×D×inc+inc) + full class_means (C×D) + full class_covs (C×D×D)."""
    # Per-task classifier heads (SimpleContinualLinear): each head D×inc + inc (bias)
    per_head = D * INC + INC
    heads = T * per_head  # 10 × (768×20+20) = 153,800
    # Class means: C × D (numpy float64)
    means = C * D           # 153,600
    # Class covs: C × D × D (torch float32)
    covs = C * D * D        # 117,964,800
    return {
        "learnable": heads,
        "stored_statistics": means + covs,
        "components": {
            f"Per-task heads ({T} × 768×20)": heads,
            "class_means (200×768)": means,
            "class_covs (200×768×768)": covs,
        },
    }


def count_ssiat():
    """SSIAT: adapter(ffn=64)×12 + per-task heads(T×D×inc) + means(C×D) + full covs(C×D×D) + radius."""
    ffn = 64
    per_layer = 2 * D * ffn + ffn + D
    adapter = L * per_layer
    per_head = D * INC  # bias=False
    heads = T * per_head
    means = C * D
    covs = C * D * D
    radius = 1  # scalar
    return {
        "learnable": adapter + heads,
        "stored_statistics": means + covs + radius,
        "components": {
            f"Adapter (12 layers, ffn=64)": adapter,
            f"Per-task heads ({T} × 768×20)": heads,
            "class_means (200×768)": means,
            "class_covs (200×768×768)": covs,
        },
    }


def count_fecam():
    """FeCAM: SSF params + CosineLinear + cov_mats (list of C×D×D corrcoef)."""
    per_block = 2 * (3 * D + D + H + D + D + D)
    ssf_params = L * per_block + 2 * D  # ~205,824
    fc = D * C + 1
    cov_mats = C * D * D  # stored as list of correlation matrices
    return {
        "learnable": ssf_params + fc,
        "stored_statistics": cov_mats,
        "components": {
            f"SSF (12 blocks)": ssf_params,
            "CosineLinear (768×200)": fc,
            "cov_mats (200 × 768×768 corrcoef)": cov_mats,
        },
    }


def count_cofima():
    """COFiMA: SLCA head + means + covs + prev_nets + init_nets + fisher_mat."""
    per_head = D * INC + INC
    heads = T * per_head
    means = C * D
    covs = C * D * D
    # Model snapshots: ~T copies of full ViT-B state_dict
    # Actually the paper stores trainable params only (fc.heads + backbone params)
    # Let's estimate: per task ~ all trainable + full model snapshot
    # In code: prev_nets stores deepcopy of self._network.state_dict() (all params)
    # ViT-B state_dict is ~85.8M values → 9 copies for tasks 1-9
    n_snapshots = T - 1
    snapshots = n_snapshots * VIT_B_PARAMS  # 9 × 85.8M = 772M
    # Fisher: diag per param, stored as dict of tensors → same size as state_dict
    fisher = (T - 1) * VIT_B_PARAMS  # approximate (Fisher is per task)
    return {
        "learnable": heads,
        "stored_statistics": means + covs + snapshots + fisher,
        "components": {
            f"Per-task heads ({T} × 768×20)": heads,
            "class_means (200×768)": means,
            "class_covs (200×768×768)": covs,
            f"Model snapshots ({n_snapshots} × ViT-B)": snapshots,
            f"Fisher matrices ({n_snapshots} × ViT-B)": fisher,
        },
    }


def count_mos():
    """MOS: adapter_list(T×per_task_adapter) + cur_adapter + adapter_ema + CosineLinear + cls_mean + cls_cov(diag)."""
    ffn = 16
    per_layer = 2 * D * ffn + ffn + D  # 25,360
    per_task_adapter = L * per_layer    # 304,320
    # adapter_list: T copies
    adapter_list = T * per_task_adapter  # 10 × 304,320
    # cur_adapter: 1 copy
    cur_adapter = L * per_layer
    # adapter_ema: 1 copy (buffer/non-learnable)
    adapter_ema = L * per_layer
    # CosineLinear
    fc = D * C + 1
    # cls_mean (dict): C × D
    means = C * D
    # cls_cov (variance mode, dict): C × D (diagonal only)
    covs = C * D
    learnable = adapter_list + cur_adapter + fc
    stored = adapter_ema + means + covs
    return {
        "learnable": learnable,
        "stored_statistics": stored,
        "components": {
            f"adapter_list ({T} tasks × adapter)": adapter_list,
            "cur_adapter (trainable)": cur_adapter,
            "adapter_ema (momentum buffer)": adapter_ema,
            "CosineLinear (768×200)": fc,
            "cls_mean (200×768)": means,
            "cls_cov variance (200×768)": covs,
        },
    }


def count_tuna():
    """TUNA: adapter_list(T) + cur_adapter + merged_adapter + fc + fc_shared_cls + expert_calib + cls_mean + cls_cov."""
    ffn = 16
    per_layer = 2 * D * ffn + ffn + D
    per_task_adapter = L * per_layer
    adapter_list = T * per_task_adapter
    cur_adapter = L * per_layer
    merged_adapter = L * per_layer  # buffer
    # fc: TunaLinear per-task heads: T × D × inc
    per_head = D * INC
    fc = T * per_head  # bias=False
    # fc_shared_cls: TunaLinear same size
    fc_shared = T * per_head
    # expert_calibration: T × 2 (scale + bias per task)
    expert_calib = T * 2
    # cls_mean + cls_cov (variance)
    means = C * D
    covs = C * D
    learnable = adapter_list + cur_adapter + fc + fc_shared + expert_calib
    stored = merged_adapter + means + covs
    return {
        "learnable": learnable,
        "stored_statistics": stored,
        "components": {
            f"adapter_list ({T} tasks × adapter)": adapter_list,
            "cur_adapter (trainable)": cur_adapter,
            "merged_adapter (buffer)": merged_adapter,
            f"Per-task heads ({T} × 768×20)": fc,
            f"fc_shared_cls ({T} × 768×20)": fc_shared,
            "expert_calibration": expert_calib,
            "cls_mean (200×768)": means,
            "cls_cov variance (200×768)": covs,
        },
    }


def count_ease():
    """EASE: cur_adapter(f=16)×12 + adapter_list(T-1) + fc(multi-adapter concat) + proxy_fc."""
    ffn = 16
    per_layer = 2 * D * ffn + ffn + D  # 25,360
    cur_adapter = L * per_layer  # 304,320
    # adapter_list: (T-1) historical copies
    adapter_list = (T - 1) * per_layer * L  # 9 × 304,320 = 2,738,880
    # fc: EaseCosineLinear with init_ptm, num_adapters=1
    # dims: concat of [init_ptm(768) + T * adapters] → total_dim = 768 + 10*768 = 8448
    # But looking at the code, EASE constructs fc per class with variable num_adapters
    # Simplified: fc stores up to C × [(1 init + T adapters) × D]
    n_sources = 1 + T  # 1 init PTM + T per-task adapters
    fc = C * n_sources * D  # 200 × 11 × 768 = 1,689,600
    fc_bias = 1
    # proxy_fc: D × inc + 1 (train-time only)
    proxy_fc = D * INC + 1
    learnable = cur_adapter + fc + fc_bias + proxy_fc
    stored_historical = adapter_list
    return {
        "learnable": learnable,
        "stored_statistics": stored_historical,
        "components": {
            "cur_adapter (trainable)": cur_adapter,
            f"adapter_list ({T-1} historical copies)": adapter_list,
            f"EaseCosineLinear (200 × 11 × 768)": fc,
            "proxy_fc (768×20)": proxy_fc,
        },
    }


def count_spie(mode="variance", lowrank_rank=0):
    """SPIE: Shared LoRA + Expert VeRA + expert_tokens + adapter history + Vera projections + classifier heads + statistical storage.
    
    Note: Both Shared LoRA and Expert VeRA are only applied to MLP (fc1/fc2), NOT to attention (qkv/proj).
    Reference: vit_spie.py MLPMixedAdapterBlock.forward() lines 166-219.
    """
    shared_lora_rank = 8
    vera_rank = 256
    expert_tokens_n = 4

    # ── Shared LoRA (r=8): MLP only, fc1 + fc2 per layer ──
    # fc1: down(D×r) + up(r×H) = 768×8 + 8×3072 = 6,144 + 24,576 = 30,720
    # fc2: down(H×r) + up(r×D) = 3072×8 + 8×768 = 24,576 + 6,144 = 30,720
    per_layer_shared = (D * shared_lora_rank + shared_lora_rank * H) + (H * shared_lora_rank + shared_lora_rank * D)
    # = 30,720 + 30,720 = 61,440
    shared_lora = L * per_layer_shared  # 12 * 61,440 = 737,280

    # ── Expert VeRA learnable (r=256): MLP only, fc1 + fc2 per layer ──
    # VeraLinear has lambda_d(rank) + lambda_b(out_features)
    # fc1: lambda_d(256) + lambda_b(3072) = 3,328
    # fc2: lambda_d(256) + lambda_b(768) = 1,024
    per_layer_expert_learnable = (vera_rank + H) + (vera_rank + D)  # 3,328 + 1,024 = 4,352
    expert_learnable = L * per_layer_expert_learnable  # 12 * 4,352 = 52,224

    # ── VeRA frozen projections (fc1+fc2 only) ──
    # These are SVD-derived A/B matrices from already-stored pretrained fc1/fc2 weights.
    # They contain NO new information – can be recomputed on-the-fly and released.
    # Marked separately as "recomputable" for fair comparison.
    per_layer_vera_buffer = (vera_rank * D + H * vera_rank) + (vera_rank * H + D * vera_rank)
    vera_buffers = L * per_layer_vera_buffer  # 12 * 1,966,080 = 23,592,960
    vera_buffers_recomputable = vera_buffers  # flag as recomputable from backbone weights

    # ── Expert tokens: expert_tokens × D (learnable per task) ──
    expert_tokens_cur = expert_tokens_n * D  # 4*768 = 3,072
    # Historical: (T-1) × expert_tokens × D
    expert_tokens_history = (T - 1) * expert_tokens_n * D  # 9*3,072 = 27,648

    # ── Adapter history: (T-1) × L × per_layer_expert_learnable ──
    # Each stored adapter (MLPVeRAAdapter) has the same params as a current adapter
    adapter_history = (T - 1) * L * per_layer_expert_learnable  # 9 * 12 * 4,352 = 470,016

    # ── Classifier heads ──
    # fc_shared_cls: TunaLinear, T × D × inc (bias=False)
    per_head = D * INC
    fc_shared = T * per_head  # 10 * 15,360 = 153,600
    # expert_heads: TunaLinear, T × 2D × inc (bias=False)
    per_expert_head = 2 * D * INC
    expert_heads = T * per_expert_head  # 10 * 30,720 = 307,200

    # ── Statistical storage ──
    means = C * D  # 200 × 768 = 153,600

    if mode == "variance":
        cov_storage = C * D  # 153,600
    elif mode == "diag_lowrank":
        # per class: diag(D) + basis(D×r) + values(r)
        r = lowrank_rank
        cov_storage = C * (D + D * r + r)  # 200 * (768 + 768*r + r)
    else:
        cov_storage = C * D * D  # full covariance

    # Total
    learnable = (shared_lora + expert_learnable + expert_tokens_cur +
                 fc_shared + expert_heads)
    # Historical + statistical storage that MUST persist
    stored_persistent = (expert_tokens_history + adapter_history + means + cov_storage)
    # VeRA projections: recomputable from backbone weights, can be released
    stored_recomputable = vera_buffers
    # Total stored including recomputable
    stored_total = stored_persistent + stored_recomputable

    components = [
        (f"Shared LoRA (r=8, 12 layers)", shared_lora, "learnable"),
        (f"Expert VeRA learnable (r=256, 12 layers)", expert_learnable, "learnable"),
        ("Expert tokens (current, 4×768)", expert_tokens_cur, "learnable"),
        (f"VeRA projections (r=256, 12 layers) [recomputable]", vera_buffers, "recomputable"),
        (f"Expert token history ({T-1} tasks)", expert_tokens_history, "stored"),
        (f"Expert VeRA history ({T-1} tasks)", adapter_history, "stored"),
        (f"fc_shared_cls ({T} × 768×20)", fc_shared, "learnable"),
        (f"expert_heads ({T} × 1536×20)", expert_heads, "learnable"),
        (f"shared_cls_mean ({C}×768)", means, "stored"),
        (f"shared_cls_cov ({mode}, C={C})", cov_storage, "stored"),
    ]

    return {
        "learnable": learnable,
        "stored_statistics": stored_total,
        "stored_persistent": stored_persistent,
        "stored_recomputable": stored_recomputable,
        "components": components,
    }


# ══════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════

METHODS = OrderedDict([
    ("L2P", count_l2p),
    ("DualPrompt", count_dualprompt),
    ("CODA-Prompt", count_coda_prompt),
    ("APER (VPT)", count_aper_vpt),
    ("APER (SSF)", count_aper_ssf),
    ("APER (adapter)", count_aper_adapter),
    ("SLCA", count_slca),
    ("SSIAT", count_ssiat),
    ("FeCAM", count_fecam),
    ("RanPAC", count_ranpac),
    ("ACIL", count_acil),
    ("MiN", count_min),
    ("MOS (variance)", count_mos),
    ("TUNA (variance)", count_tuna),
    ("EASE", count_ease),
    ("COFiMA", count_cofima),
    ("SPIE (variance)", lambda: count_spie("variance")),
    ("SPIE (lowrank r=4)", lambda: count_spie("diag_lowrank", 4)),
    ("SPIE (lowrank r=8)", lambda: count_spie("diag_lowrank", 8)),
    ("SPIE (lowrank r=16)", lambda: count_spie("diag_lowrank", 16)),
    ("SPIE (lowrank r=32)", lambda: count_spie("diag_lowrank", 32)),
])


# ══════════════════════════════════════════════════════════════════════════
# Display
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("  SPIE Codebase – Comprehensive Parameter Count Analysis (Analytical)")
    print(f"  ViT-B/16 backbone: {VIT_B_PARAMS:,} params (~85.8M), D=768, L=12")
    print(f"  DomainNet: C=200 classes, T=10 tasks (init=20, inc=20)")
    print("=" * 90)

    results = []
    for name, fn in METHODS.items():
        print(f"\n{'─'*80}")
        print(f"  {name}")
        print(f"{'─'*80}")

        result = fn()
        results.append((name, result))

        total = result["learnable"] + result.get("stored_statistics", 0)
        recomputable = result.get("stored_recomputable", 0)
        persistent_total = total - recomputable

        # Display components with category tags
        print(f"  {'Component':<55s} {'Numel':>12s}  Category")
        print(f"  {'─'*55} {'─'*12}  {'─'*15}")
        if isinstance(result["components"], list):
            for comp_name, count, cat in result["components"]:
                tag = {"learnable": "[trainable]", "stored": "[persistent]", "recomputable": "[recomputable]"}.get(cat, "")
                print(f"  {comp_name:<55s} {count:>12,}  {tag}")
        else:
            for comp_name, count in result["components"].items():
                print(f"  {comp_name:<55s} {count:>12,}")

        print(f"  {'─'*67}")
        print(f"  {'Learnable params:':<55s} {result['learnable']:>12,}  ({_fmt(result['learnable'])})")
        stored = result.get("stored_statistics", 0)
        if stored > 0:
            print(f"  {'Stored (all, incl. recomputable):':<55s} {stored:>12,}  ({_fmt(stored)})")
        if recomputable > 0:
            print(f"  {'  of which recomputable:':<55s} {recomputable:>12,}  ({_fmt(recomputable)})")
            print(f"  {'  persistent stored only:':<55s} {stored - recomputable:>12,}  ({_fmt(stored - recomputable)})")
            print(f"  {'TOTAL (persistent = learn+stored):':<55s} {persistent_total:>12,}  ({_fmt(persistent_total)} = {_pct(persistent_total)})")
        print(f"  {'TOTAL additional (all):':<55s} {total:>12,}  ({_fmt(total)} = {_pct(total)})")
        print(f"  {'Backbone (ViT-B):':<55s} {VIT_B_PARAMS:>12,}  ({_fmt(VIT_B_PARAMS)})")
        print(f"  {'System total:':<55s} {total + VIT_B_PARAMS:>12,}  ({_fmt(total + VIT_B_PARAMS)})")

    # ── Summary table ──
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY TABLE")
    print(f"  Backbone (ViT-B): {VIT_B_PARAMS:,} ({_fmt(VIT_B_PARAMS)})")
    print(f"{'='*110}")
    header = f"{'Method':<24s} {'Learnable':>10s} {'Stored':>10s} {'Recomput.':>10s} {'Persistent':>10s} {'% BB':>7s} {'Category'}"
    print(header)
    print(f"{'─'*24} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*7} {'─'*20}")

    # Determine category
    def categorize(name, total, stored):
        ratio = total / VIT_B_PARAMS
        if ratio > 10:
            return "🔥 Heavy (>10×BB)"
        elif ratio > 1:
            return "⚡ Large (1-10×BB)"
        elif ratio > 0.1:
            return "◆ Medium (~10%BB)"
        elif ratio > 0.01:
            return "○ Light (~1%BB)"
        else:
            return "  Minimal"

    for name, r in sorted(results, key=lambda x: x[1]["learnable"] + x[1].get("stored_statistics", 0)):
        learnable = r["learnable"]
        stored = r.get("stored_statistics", 0)
        recomputable = r.get("stored_recomputable", 0)
        persistent = learnable + (stored - recomputable)
        total = learnable + stored
        pct = total / VIT_B_PARAMS * 100
        cat = categorize(name, total, stored)
        print(f"{name:<24s} {learnable:>10,} {stored:>10,} {recomputable:>10,} {persistent:>10,} {pct:>6.1f}% {cat}")

    print(f"\n{'─'*80}")
    print(f"Note: 'Recomput.' = can be derived from frozen backbone weights on-the-fly.")
    print(f"      'Persistent' = Learnable + (Stored - Recomputable) = must-persist count.")
    print(f"      For SPIE, VeRA projections ({_fmt(23592960)}) are SVD of backbone MLP weights –")
    print(f"      they contain NO new information and can be released between training/inference.")


if __name__ == "__main__":
    main()
