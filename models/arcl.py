import logging
from typing import List

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import SimpleContinualLinear
from backbone.vit_arcl import (
    pretrained_vit_b16_224_arcl,
    pretrained_vit_b16_224_in21k_arcl,
    vit_base_patch16_224_arcl,
    vit_base_patch16_224_in21k_arcl,
)
from models.base import BaseLearner
from utils.mod_adam_arcl import ARCLModAdam
from utils.toolkit import tensor2numpy

try:
    from torch.amp import GradScaler as TorchGradScaler
    from torch.amp import autocast as torch_autocast

    def _make_grad_scaler(enabled):
        return TorchGradScaler("cuda", enabled=enabled)

    def _autocast_context(enabled):
        return torch_autocast("cuda", dtype=torch.float16, enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler as TorchGradScaler
    from torch.cuda.amp import autocast as torch_autocast

    def _make_grad_scaler(enabled):
        return TorchGradScaler(enabled=enabled)

    def _autocast_context(enabled):
        return torch_autocast(dtype=torch.float16, enabled=enabled)


def _get_optimizer(name, params, lr, weight_decay):
    if name == "mod_adam":
        return ARCLModAdam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def _get_scheduler(name, optimizer, epochs, min_lr):
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=min_lr)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=epochs, gamma=0.1)
    if name == "multistep":
        return None
    if name == "constant":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def find_split_point(data: np.ndarray, eps=1e-3):
    sorted_x = np.sort(data)
    sorted_y = np.arange(len(data))
    regular_x = np.linspace(sorted_x.min(), sorted_x.max(), 10000)
    regular_y = np.interp(regular_x, sorted_x, sorted_y)
    regular_y = gaussian_filter(regular_y, sigma=20)
    first_derivative = np.gradient(regular_y, regular_x)
    second_derivative = np.gradient(first_derivative, regular_x)
    min_idx = np.argmin(second_derivative)
    return min(max(regular_x[min_idx], regular_x[0]) - eps, regular_x[-1])


class ARCLNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone_type = args["backbone_type"].lower()
        pretrained = bool(args.get("pretrained", True))
        if backbone_type in {"vit_base_patch16_224_arcl", "pretrained_vit_b16_224_arcl"}:
            self.backbone = vit_base_patch16_224_arcl(pretrained=pretrained)
            self.pretrained_source = "vit_base_patch16_224"
        elif backbone_type in {"vit_base_patch16_224_in21k_arcl", "pretrained_vit_b16_224_in21k_arcl"}:
            self.backbone = vit_base_patch16_224_in21k_arcl(pretrained=pretrained)
            self.pretrained_source = "vit_base_patch16_224_in21k"
        else:
            raise ValueError(f"Unsupported ARCL backbone: {args['backbone_type']}")

        self.fc = None
        self.out_dim = self.backbone.out_dim

    @property
    def feature_dim(self):
        return self.out_dim

    def update_fc(self, nb_new_classes):
        if self.fc is None:
            self.fc = SimpleContinualLinear(self.feature_dim, nb_new_classes)
        else:
            self.fc.update(nb_new_classes, freeze_old=True)

    def extract_vector(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        out = self.fc(feats)
        out["features"] = feats
        return out


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = ARCLNet(args)
        self.args = args
        self.batch_size = int(args.get("batch_size", 100))
        self.num_workers = int(args.get("num_workers", 8))
        self.epochs = int(args.get("epochs", args.get("tuned_epoch", 5)))
        self.lr = float(args.get("lr", args.get("init_lr", 1e-2)))
        self.weight_decay = float(args.get("weight_decay", 5e-5))
        self.min_lr = float(args.get("min_lr", 1e-5))
        self.optimizer_type = args.get("optimizer", "mod_adam").lower()
        self.scheduler_type = args.get("lr_sch", args.get("scheduler", "multistep")).lower()
        self.temperature = float(args.get("temperature", 30.0))
        self.use_update_scaling = bool(args.get("use_update_scaling", True))
        self.use_amp = bool(args.get("use_amp", True))
        self.lr_scale = float(args.get("lr_scale", 0.01))
        self.lr_scale_patterns = tuple(args.get("lr_scale_patterns", ["qkv"]))
        self.decay_milestones = list(args.get("decay_milestones", [5, 8]))
        self.decay_rate = float(args.get("decay_rate", 0.1))

        self.old_avg_attn_map = None
        self.avg_attn_map = None
        self.temp_count = 0

    @staticmethod
    def _tensor_stats(name, tensor):
        tensor = tensor.detach()
        finite_mask = torch.isfinite(tensor)
        finite_count = int(finite_mask.sum().item())
        total_count = tensor.numel()
        if finite_count == 0:
            return f"{name}: finite=0/{total_count}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}"

        finite_values = tensor[finite_mask]
        return (
            f"{name}: finite={finite_count}/{total_count}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
            f"min={float(finite_values.amin().item()):.4f}, max={float(finite_values.amax().item()):.4f}"
        )

    def _assert_finite_parameters(self):
        bad_params = []
        for name, param in self._network.named_parameters():
            if not torch.isfinite(param).all():
                bad_params.append(self._tensor_stats(name, param))
                if len(bad_params) >= 8:
                    break
        if bad_params:
            raise FloatingPointError("Non-finite ARCL parameters before forward: " + " | ".join(bad_params))

    def _diagnose_nonfinite_forward(self, inputs):
        diagnostics = [self._tensor_stats("inputs", inputs)]
        network = self._network
        network.eval()

        with torch.no_grad():
            x = inputs.float()
            x = network.backbone.patch_embed(x)
            diagnostics.append(self._tensor_stats("patch_embed", x))
            if not torch.isfinite(x).all():
                return " | ".join(diagnostics)

            cls_tokens = network.backbone.cls_token.expand(x.shape[0], -1, -1).float()
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + network.backbone.pos_embed.float()
            x = network.backbone.pos_drop(x)
            diagnostics.append(self._tensor_stats("pos_drop", x))
            if not torch.isfinite(x).all():
                return " | ".join(diagnostics)

            for block_idx, block in enumerate(network.backbone.blocks):
                x = block(x)
                diagnostics.append(self._tensor_stats(f"block_{block_idx}", x))
                if not torch.isfinite(x).all():
                    diagnostics.append(self._tensor_stats(f"block_{block_idx}_q_proj_weight", block.attn.q_proj.weight))
                    diagnostics.append(self._tensor_stats(f"block_{block_idx}_k_proj_weight", block.attn.k_proj.weight))
                    diagnostics.append(self._tensor_stats(f"block_{block_idx}_v_proj_weight", block.attn.v_proj.weight))
                    return " | ".join(diagnostics)

            x = network.backbone.norm(x)
            diagnostics.append(self._tensor_stats("backbone_norm", x))
            feats = x[:, 0]
            diagnostics.append(self._tensor_stats("features_fp32", feats))
            current_head = network.fc.heads[self._cur_task]
            current_logits = current_head(feats)
            diagnostics.append(self._tensor_stats("current_head_logits_fp32", current_logits))

        return " | ".join(diagnostics)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(data_manager.get_task_size(self._cur_task))

        if self._cur_task == 0:
            logging.info(
                "ARCL pretrained source aligned with other methods: %s (pretrained=%s)",
                self._network.pretrained_source,
                bool(self.args.get("pretrained", True)),
            )

        logging.info("Learning on %d-%d", self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train"
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self._train_task(self.train_loader)

    def _train_task(self, train_loader):
        self._network.to(self._device)
        self._set_trainable_state()
        self._assert_finite_parameters()

        qkv_params = []
        for name, param in self._network.backbone.named_parameters():
            if any(token in name for token in ("q_proj.weight", "k_proj.weight", "v_proj.weight")):
                param.requires_grad_(True)
                qkv_params.append(param)
            else:
                param.requires_grad_(False)

        for task_id, head in enumerate(self._network.fc.heads):
            head.requires_grad_(task_id == self._cur_task)

        head_params = [p for p in self._network.fc.heads[self._cur_task].parameters() if p.requires_grad]
        param_groups = self._build_param_groups(qkv_params, head_params)
        optimizer = _get_optimizer(self.optimizer_type, param_groups, self.lr, self.weight_decay)
        scheduler = _get_scheduler(self.scheduler_type, optimizer, self.epochs, self.min_lr)
        criterion = nn.CrossEntropyLoss()
        scaler = _make_grad_scaler(enabled=self.use_amp and torch.cuda.is_available())

        if self.avg_attn_map is None:
            num_layers = len(self._network.backbone.blocks)
            num_patches = self._network.backbone.patch_embed.num_patches
            self.avg_attn_map = torch.zeros(num_layers, num_patches, device=self._device)

        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
            if self.scheduler_type == "multistep":
                self._step_multistep_scheduler(optimizer, epoch)
            elif scheduler is not None and epoch == 0:
                scheduler.step(0)

            losses = 0.0
            correct, total = 0, 0
            rollout_batch_maps = []

            for batch_indices, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                if not torch.isfinite(inputs).all():
                    raise FloatingPointError(
                        "Non-finite ARCL inputs detected: "
                        + self._tensor_stats("inputs", inputs)
                        + f" | batch_indices={tensor2numpy(batch_indices).tolist()} | targets={tensor2numpy(targets).tolist()}"
                    )
                optimizer.zero_grad()
                with _autocast_context(enabled=self.use_amp and torch.cuda.is_available()):
                    logits = self._network(inputs)["logits"]
                    task_logits = logits[:, self._known_classes : self._total_classes]
                loss = criterion(task_logits.float() / self.temperature, targets - self._known_classes)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite loss detected in ARCL "
                        f"(task={self._cur_task}, epoch={epoch + 1}) | "
                        + self._tensor_stats("task_logits_amp", task_logits)
                        + " | "
                        + self._diagnose_nonfinite_forward(inputs)
                        + f" | batch_indices={tensor2numpy(batch_indices).tolist()} | targets={tensor2numpy(targets).tolist()}"
                    )

                scaler.scale(loss).backward()
                update_factor_map = self._build_update_factor_map(inputs.shape[0]) if self._cur_task > 0 else {}

                if isinstance(optimizer, ARCLModAdam):
                    scaler.unscale_(optimizer)
                    optimizer.step(update_factor_map=update_factor_map if self.use_update_scaling else None)
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                losses += loss.item()
                preds = task_logits.argmax(dim=1) + self._known_classes
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

                if epoch == self.epochs - 1:
                    rollout_batch_maps.extend(self._collect_rollout_maps(targets))

            if scheduler is not None:
                scheduler.step(epoch + 1)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / len(train_loader)
            lr = optimizer.param_groups[0]["lr"]
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(
                f"Task {self._cur_task}, Epoch {epoch + 1}/{self.epochs} => Loss {avg_loss:.3f}, Train_accy {train_acc:.2f}"
            )

            if epoch == self.epochs - 1:
                self._update_attention_statistics(rollout_batch_maps)

    def _set_trainable_state(self):
        self._network.eval()
        self._network.backbone.eval()
        for block in self._network.backbone.blocks:
            block.attn.train()
        for task_id, head in enumerate(self._network.fc.heads):
            if task_id == self._cur_task:
                head.train()
            else:
                head.eval()

    def _build_param_groups(self, qkv_params, head_params):
        scaled_patterns = set()
        for pattern in self.lr_scale_patterns:
            if pattern == "qkv":
                scaled_patterns.update({"q_proj.weight", "k_proj.weight", "v_proj.weight"})
            else:
                scaled_patterns.add(pattern)

        qkv_aliases = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "q_proj", "k_proj", "v_proj"}
        qkv_lr = self.lr * self.lr_scale if any(pattern in qkv_aliases for pattern in scaled_patterns) else self.lr
        return [
            {"params": head_params, "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": qkv_params, "lr": qkv_lr, "weight_decay": self.weight_decay},
        ]

    def _step_multistep_scheduler(self, optimizer, epoch):
        lr_scale = self.decay_rate ** sum(epoch >= milestone for milestone in self.decay_milestones)
        optimizer.param_groups[0]["lr"] = self.lr * lr_scale
        optimizer.param_groups[1]["lr"] = self.lr * self.lr_scale * lr_scale

    def _build_update_factor_map(self, batch_size):
        if self.old_avg_attn_map is None:
            return {}

        update_factor_map = {}
        cls_prefix = torch.zeros(1, device=self._device, dtype=self.old_avg_attn_map.dtype)

        for block_idx, block in enumerate(self._network.backbone.blocks):
            layer_mask = torch.cat([cls_prefix, self.old_avg_attn_map[block_idx]], dim=0).view(1, 1, -1, 1)
            attn = block.attn.attn.detach()
            grad_attn = block.attn.attn_no_softmax.grad.detach()
            k = block.attn.k
            q_scaled = block.attn.q_scaled
            inp = block.attn.input
            out_grad = block.attn.out.grad.detach()

            attn = attn * layer_mask.to(attn.dtype)
            grad_attn = grad_attn * layer_mask.to(grad_attn.dtype)

            act_dtype = k.dtype
            inp_flat = inp.to(act_dtype).reshape(batch_size * inp.shape[1], -1)
            grad_attn = grad_attn.to(act_dtype)
            k = k.to(act_dtype)
            q_scaled = q_scaled.to(act_dtype)

            q_grad = (grad_attn @ k) * block.attn.scale
            q_grad = q_grad.transpose(1, 2).reshape(batch_size * inp.shape[1], -1)
            q_weight_grad = (q_grad.T @ inp_flat).to(block.attn.q_proj.weight.grad.dtype)

            k_grad = grad_attn.transpose(-2, -1) @ q_scaled
            k_grad = k_grad.transpose(1, 2).reshape(batch_size * inp.shape[1], -1)
            k_weight_grad = (k_grad.T @ inp_flat).to(block.attn.k_proj.weight.grad.dtype)

            attn = attn.to(out_grad.dtype)
            inp_flat_v = inp.to(out_grad.dtype).reshape(batch_size * inp.shape[1], -1)
            v_grad = (out_grad.transpose(-2, -1) @ attn).transpose(-2, -1)
            v_grad = v_grad.transpose(1, 2).reshape(batch_size * inp.shape[1], -1)
            v_weight_grad = (v_grad.T @ inp_flat_v).to(block.attn.v_proj.weight.grad.dtype)

            update_factor_map[id(block.attn.q_proj.weight)] = self._safe_ratio(q_weight_grad, block.attn.q_proj.weight.grad)
            update_factor_map[id(block.attn.k_proj.weight)] = self._safe_ratio(k_weight_grad, block.attn.k_proj.weight.grad)
            update_factor_map[id(block.attn.v_proj.weight)] = self._safe_ratio(v_weight_grad, block.attn.v_proj.weight.grad)

        return update_factor_map

    @staticmethod
    def _safe_ratio(new_grad, old_grad):
        ratio = new_grad / old_grad.detach()
        ratio = torch.where(torch.isnan(ratio), torch.ones_like(ratio), ratio)
        ratio = torch.where(torch.isinf(ratio), torch.ones_like(ratio), ratio)
        return ratio.clamp(-10, 10).detach()

    def _collect_rollout_maps(self, targets):
        maps = []
        num_layers = len(self._network.backbone.blocks)
        for sample_idx in range(self._network.backbone.blocks[0].attn.attn_clone.shape[0]):
            attention_map = torch.stack(
                [self._network.backbone.blocks[layer_idx].attn.attn_clone[sample_idx].detach() for layer_idx in range(num_layers)]
            )
            attention_map = attention_map.mean(dim=1)
            residual = torch.eye(attention_map.size(-1), device=attention_map.device)
            aug = attention_map + residual
            aug = aug / aug.sum(dim=-1, keepdim=True)
            rollout = torch.zeros_like(aug)
            rollout[0] = aug[0]
            for layer_idx in range(1, aug.size(0)):
                rollout[layer_idx] = aug[layer_idx] @ rollout[layer_idx - 1]
            maps.append((int(targets[sample_idx].item()), rollout[:, 0, 1:].detach()))
        return maps

    def _update_attention_statistics(self, rollout_maps):
        if not rollout_maps:
            return

        data_for_all_class = {}
        all_rollout_attentions = {}
        for target, rollout in rollout_maps:
            if target not in data_for_all_class:
                data_for_all_class[target] = torch.tensor([], device=self._device)
                all_rollout_attentions[target] = torch.zeros((0, rollout.shape[0], rollout.shape[1]), device=self._device)
            all_rollout_attentions[target] = torch.cat(
                (all_rollout_attentions[target], rollout.reshape(1, rollout.shape[0], rollout.shape[1])),
                dim=0,
            )
            data_for_all_class[target] = torch.cat((data_for_all_class[target], rollout.flatten()))

        for classid in data_for_all_class.keys():
            data = data_for_all_class[classid].detach().cpu().numpy()
            data_windows = sliding_window_view(data.reshape(data.shape[0]), rollout_maps[0][1].numel())
            image_stack = all_rollout_attentions[classid]
            for sample_idx in range(int(data.shape[0] / rollout_maps[0][1].numel())):
                data_sample = data_windows[sample_idx].reshape(image_stack.shape[1], image_stack.shape[2])
                image_sample = image_stack[sample_idx]
                split_points = [find_split_point(data_sample[layer_idx]) for layer_idx in range(data_sample.shape[0])]

                binary_mask = []
                for layer_idx in range(data_sample.shape[0]):
                    binary_mask.append(
                        torch.where(
                            image_sample[layer_idx] <= split_points[layer_idx],
                            torch.tensor(1.0, device=self._device),
                            torch.tensor(0.0, device=self._device),
                        )
                    )

                self.avg_attn_map += torch.stack(binary_mask, dim=0)
                self.temp_count += 1

        self.old_avg_attn_map = (self.avg_attn_map / max(self.temp_count, 1)).detach().clone()
