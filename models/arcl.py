import logging
from typing import List

import numpy as np
import torch
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


def _get_optimizer(name, params, lr, weight_decay):
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
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[70, 100], gamma=0.1)
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
        self.lr = float(args.get("lr", 1e-4))
        self.head_lr = float(args.get("head_lr", 1e-2))
        self.weight_decay = float(args.get("weight_decay", 5e-5))
        self.min_lr = float(args.get("min_lr", 1e-5))
        self.optimizer_type = args.get("optimizer", "adam").lower()
        self.scheduler_type = args.get("scheduler", "cosine").lower()
        self.temperature = float(args.get("temperature", 1.0))
        self.use_update_scaling = bool(args.get("use_update_scaling", True))

        self.old_avg_attn_map = None
        self.avg_attn_map = None
        self.temp_count = 0

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
        self._network.train()

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
        optimizer = ARCLModAdam(
            [
                {"params": qkv_params, "lr": self.lr, "weight_decay": self.weight_decay},
                {"params": head_params, "lr": self.head_lr, "weight_decay": self.weight_decay},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = _get_scheduler(self.scheduler_type, optimizer, self.epochs, self.min_lr)
        criterion = nn.CrossEntropyLoss()

        if self.avg_attn_map is None:
            num_layers = len(self._network.backbone.blocks)
            num_patches = self._network.backbone.patch_embed.num_patches
            self.avg_attn_map = torch.zeros(num_layers, num_patches, device=self._device)

        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
            losses = 0.0
            correct, total = 0, 0
            rollout_batch_maps: List[torch.Tensor] = []

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                logits = self._network(inputs)["logits"]
                task_logits = logits[:, self._known_classes : self._total_classes]
                loss = criterion(task_logits / self.temperature, targets - self._known_classes)

                optimizer.zero_grad()
                loss.backward()
                update_factor_map = self._build_update_factor_map(inputs.shape[0]) if self._cur_task > 0 else {}
                optimizer.step(update_factor_map=update_factor_map if self.use_update_scaling else None)

                if self._cur_task == 0 or not self.use_update_scaling:
                    pass

                losses += loss.item()
                preds = task_logits.argmax(dim=1) + self._known_classes
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

                if epoch == self.epochs - 1:
                    rollout_batch_maps.extend(self._collect_rollout_maps())

            if scheduler is not None:
                scheduler.step()

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

    def _build_update_factor_map(self, batch_size):
        if self.old_avg_attn_map is None:
            return {}

        update_factor_map = {}
        cls_prefix = torch.zeros(1, device=self._device, dtype=self.old_avg_attn_map.dtype)

        for block_idx, block in enumerate(self._network.backbone.blocks):
            layer_mask = torch.cat([cls_prefix, self.old_avg_attn_map[block_idx]], dim=0).view(1, 1, -1, 1)

            attn = block.attn.attn.detach()
            grad_attn = block.attn.attn_no_softmax.grad.detach()
            attn = attn * layer_mask
            grad_attn = grad_attn * layer_mask

            q = block.attn.q
            k = block.attn.k
            q_scaled = block.attn.q_scaled
            inp = block.attn.input
            out_grad = block.attn.out.grad.detach()

            q_grad = (grad_attn @ k) * block.attn.scale
            q_grad = q_grad.transpose(1, 2).reshape(batch_size * inp.shape[1], -1)
            q_weight_grad = q_grad.T @ inp.reshape(batch_size * inp.shape[1], -1)

            k_grad = grad_attn.transpose(-2, -1) @ q_scaled
            k_grad = k_grad.transpose(1, 2).reshape(batch_size * inp.shape[1], -1)
            k_weight_grad = k_grad.T @ inp.reshape(batch_size * inp.shape[1], -1)

            v_grad = (out_grad.transpose(-2, -1) @ attn.to(out_grad.dtype)).transpose(-2, -1)
            v_grad = v_grad.transpose(1, 2).reshape(batch_size * inp.shape[1], -1)
            v_weight_grad = v_grad.T @ inp.reshape(batch_size * inp.shape[1], -1)

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

    def _collect_rollout_maps(self):
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
            maps.append(rollout[:, 0, 1:].detach())
        return maps

    def _update_attention_statistics(self, rollout_maps):
        if not rollout_maps:
            return

        for rollout in rollout_maps:
            rollout_np = rollout.detach().cpu().numpy()
            binary_mask = []
            for layer_map in rollout_np:
                threshold = find_split_point(layer_map)
                binary_mask.append(torch.as_tensor((layer_map <= threshold).astype(np.float32), device=self._device))
            self.avg_attn_map += torch.stack(binary_mask, dim=0)
            self.temp_count += 1

        self.old_avg_attn_map = (self.avg_attn_map / max(self.temp_count, 1)).detach().clone()
