import argparse
import copy
import json
import re
import warnings
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")

from utils.timm_compat import patch_timm_dataclass_defaults

patch_timm_dataclass_defaults()

from backbone.vit_spie_v2 import VisionTransformer


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def load_checkpoint(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def model_state(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def backbone_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefix = "backbone."
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix) and torch.is_tensor(value)
    }


def resolve_checkpoint(args: argparse.Namespace, role: str) -> Path:
    explicit = getattr(args, role)
    if explicit is not None:
        return Path(explicit)

    if args.checkpoint_dir is None:
        raise ValueError(f"Provide --{role} or --checkpoint-dir with --{role}-task.")

    task_id = getattr(args, f"{role}_task")
    if task_id is None:
        raise ValueError(f"Provide --{role}-task when using --checkpoint-dir.")
    return Path(args.checkpoint_dir) / f"task_{task_id}.pkl"


def infer_num_classes(config: Dict, *states: Dict[str, torch.Tensor]) -> int:
    if "nb_classes" in config:
        return int(config["nb_classes"])

    for state in states:
        head_weight = state.get("backbone.head.weight")
        if torch.is_tensor(head_weight):
            return int(head_weight.shape[0])

    for state in states:
        fc_head_ids = []
        for key in state:
            match = re.match(r"fc\.heads\.(\d+)\.0\.weight", key)
            if match:
                fc_head_ids.append((int(match.group(1)), state[key].shape[0]))
        if fc_head_ids:
            return int(sum(size for _, size in sorted(fc_head_ids)))

    raise ValueError("Could not infer nb_classes. Pass it in the config or use a checkpoint with backbone.head.weight.")


def build_spie_v2_backbone(config: Dict, num_classes: int, device: torch.device) -> VisionTransformer:
    backbone_type = str(config.get("backbone_type", "")).lower()
    supported = {"vit_base_patch16_224_spie_v2", "vit_base_patch16_224_in21k_spie_v2"}
    if backbone_type not in supported:
        raise ValueError(f"This script expects a SPiE v2 backbone_type in {sorted(supported)}, got {backbone_type!r}.")

    tuning_config = SimpleNamespace(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=16,
        d_model=768,
        _device=str(device),
        vpt_on=False,
        vpt_num=0,
    )
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        global_pool=False,
        drop_path_rate=0.0,
        tuning_config=tuning_config,
        r=int(config.get("r", 16)),
        expert_tokens=int(config.get("expert_tokens", 4)),
    )
    return model.to(device)


def count_indexed_modules(state: Dict[str, torch.Tensor], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)(?:\.|$)")
    indices = set()
    for key in state:
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return max(indices) + 1 if indices else 0


def prepare_backbone_structure(model: VisionTransformer, state: Dict[str, torch.Tensor]) -> None:
    adapter_count = count_indexed_modules(state, "adapter_list")
    token_count = count_indexed_modules(state, "expert_token_list")
    target_count = max(adapter_count, token_count)

    while len(model.adapter_list) < target_count or len(model.expert_token_list) < target_count:
        model.adapter_update()

    has_merged_adapter = any(key.startswith("merged_adapter.") for key in state)
    if has_merged_adapter and hasattr(model, "merged_adapter") and len(model.merged_adapter) == 0:
        model.merged_adapter = copy.deepcopy(model.cur_adapter)


def load_backbone_from_checkpoint(config: Dict, checkpoint: Dict, num_classes: int, device: torch.device) -> VisionTransformer:
    state = backbone_state(model_state(checkpoint))
    model = build_spie_v2_backbone(config, num_classes=num_classes, device=device)
    prepare_backbone_structure(model, state)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Backbone checkpoint did not match. "
            f"Missing keys: {incompatible.missing_keys}. Unexpected keys: {incompatible.unexpected_keys}."
        )
    model.eval()
    return model


def extract_features(model: VisionTransformer, inputs: torch.Tensor, expert_id: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(inputs, adapter_id=expert_id, train=False)["features"].detach().cpu()


def expert_keys(expert_id: int) -> Tuple[str, str]:
    return (f"backbone.adapter_list.{expert_id}.", f"backbone.expert_token_list.{expert_id}")


def selected_expert_tensors(state: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if torch.is_tensor(value) and any(key == prefix or key.startswith(prefix) for prefix in prefixes)
    }


def max_state_delta(source_state: Dict[str, torch.Tensor], target_state: Dict[str, torch.Tensor], expert_id: int) -> float:
    prefixes = expert_keys(expert_id)
    source = selected_expert_tensors(source_state, prefixes)
    target = selected_expert_tensors(target_state, prefixes)
    if not source:
        raise ValueError(f"No tensors found for expert_id={expert_id} in source checkpoint.")

    missing = sorted(set(source) - set(target))
    if missing:
        raise ValueError(f"Target checkpoint is missing expert tensors, first missing key: {missing[0]}")

    max_delta = 0.0
    for key, before in source.items():
        after = target[key]
        if before.shape != after.shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(before.shape)} vs {tuple(after.shape)}")
        max_delta = max(max_delta, float((before - after).abs().max().item()))
    return max_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load two trained SPiE v2 checkpoints and compare the same expert's features on a fixed random input. "
            "The feature path is exactly backbone(inputs, adapter_id=expert_id, train=False)['features']."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Training config JSON, for example exps/spie_v2_cifar.json.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory containing task_0.pkl, task_1.pkl, ...")
    parser.add_argument("--source", type=Path, help="Earlier checkpoint path, for example task_0.pkl.")
    parser.add_argument("--target", type=Path, help="Later checkpoint path, for example task_9.pkl.")
    parser.add_argument("--source-task", type=int, help="Earlier task id when using --checkpoint-dir.")
    parser.add_argument("--target-task", type=int, help="Later task id when using --checkpoint-dir.")
    parser.add_argument("--expert-id", type=int, default=0, help="Zero-based expert id. Use 0 for the first expert.")
    parser.add_argument("--batch-size", type=int, default=2, help="Random probe batch size. Default: 2.")
    parser.add_argument("--image-size", type=int, default=224, help="Random probe image size. Default: 224.")
    parser.add_argument("--seed", type=int, default=1993, help="Random probe seed. Default: 1993.")
    parser.add_argument("--device", default="cpu", help="Device used for feature extraction. Default: cpu.")
    parser.add_argument("--atol", type=float, default=0.0, help="Allowed max feature delta. Default: 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source_path = resolve_checkpoint(args, "source")
    target_path = resolve_checkpoint(args, "target")
    source_checkpoint = load_checkpoint(source_path)
    target_checkpoint = load_checkpoint(target_path)
    source_state = model_state(source_checkpoint)
    target_state = model_state(target_checkpoint)

    num_classes = infer_num_classes(config, target_state, source_state)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    probe_inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    source_backbone = load_backbone_from_checkpoint(config, source_checkpoint, num_classes, device)
    target_backbone = load_backbone_from_checkpoint(config, target_checkpoint, num_classes, device)

    source_features = extract_features(source_backbone, probe_inputs, args.expert_id)
    target_features = extract_features(target_backbone, probe_inputs, args.expert_id)
    feature_max_delta = float((source_features - target_features).abs().max().item())
    feature_mean_delta = float((source_features - target_features).abs().mean().item())
    param_max_delta = max_state_delta(source_state, target_state, args.expert_id)
    passed = feature_max_delta <= args.atol and param_max_delta <= args.atol

    print("SPiE v2 trained-checkpoint feature isolation test")
    print(f"config: {args.config}")
    print(f"source: {source_path}")
    print(f"target: {target_path}")
    print(f"expert_id: {args.expert_id} (zero-based; 0 means the first expert)")
    print(f"feature path: backbone(inputs, adapter_id={args.expert_id}, train=False)['features']")
    print(f"probe: seed={args.seed}, shape=({args.batch_size}, 3, {args.image_size}, {args.image_size})")
    print(f"feature_max_delta: {feature_max_delta:.8g}")
    print(f"feature_mean_delta: {feature_mean_delta:.8g}")
    print(f"expert_param_max_delta: {param_max_delta:.8g}")

    if passed:
        print("RESULT: PASS - watched expert features are unchanged.")
        return 0

    print("RESULT: FAIL - watched expert changed between checkpoints.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
