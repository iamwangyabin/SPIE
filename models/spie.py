import logging
from typing import Dict, List

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.vit_spie import IncrementalExpertModel
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8


def _build_task_class_counts(total_classes: int, init_cls: int, increment: int) -> List[int]:
    if init_cls <= 0:
        raise ValueError(f"init_cls must be > 0, got {init_cls}")
    if increment <= 0:
        raise ValueError(f"increment must be > 0, got {increment}")
    if total_classes < init_cls:
        raise ValueError(f"total_classes must be >= init_cls; got {total_classes} < {init_cls}")

    counts = [init_cls]
    while sum(counts) + increment < total_classes:
        counts.append(increment)
    offset = total_classes - sum(counts)
    if offset > 0:
        counts.append(offset)
    return counts


def _resolve_timm_model_name(backbone_type: str) -> str:
    name = backbone_type.lower()
    if name.endswith("_spie"):
        name = name[:-5]
    valid = {
        "vit_base_patch16_224",
        "vit_base_patch16_224_in21k",
    }
    if name not in valid:
        raise ValueError(
            f"Unsupported SPiE backbone_type '{backbone_type}'. Supported values: {sorted(valid)} "
            "with or without the '_spie' suffix."
        )
    return name


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.args["tuned_epoch"] = args["tuned_epoch"]

        self.task_class_counts = _build_task_class_counts(
            total_classes=args["nb_classes"],
            init_cls=args["init_cls"],
            increment=args["increment"],
        )
        self.task_offsets = np.cumsum([0] + self.task_class_counts[:-1]).tolist()

        self._network = IncrementalExpertModel(
            model_name=_resolve_timm_model_name(args["backbone_type"]),
            pretrained=args.get("pretrained", True),
            num_experts=len(self.task_class_counts),
            num_classes_per_expert=self.task_class_counts,
            expert_tokens=args.get("expert_tokens", 4),
            lora_rank=args.get("lora_rank", 8),
            lora_alpha=args.get("lora_alpha", 1.0),
            freeze_backbone=args.get("freeze_backbone", True),
        )

        total_params = sum(p.numel() for p in self._network.parameters())
        trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f"{total_params:,} model total parameters.")
        logging.info(f"{trainable_params:,} model trainable parameters.")

        if len(self._multiple_gpus) > 1:
            logging.info("SPiE currently uses a single device and does not wrap the model with DataParallel.")

    def after_task(self):
        self._known_classes = self._total_classes

    def _activate_task_head(self, expert_idx: int) -> None:
        for i, head in enumerate(self._network.heads.heads):
            head.requires_grad_(i == expert_idx)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        self._activate_task_head(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        del test_loader
        self._network.to(self._device)
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        self._init_train(train_loader, optimizer, scheduler)

    def get_optimizer(self):
        param_groups = self._network.parameter_groups_for_single_task(
            expert_idx=self._cur_task,
            lr=self.init_lr,
            head_lr=self.init_lr,
            expert_weight_decay=0.0,
            head_weight_decay=self.weight_decay,
        )

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9)
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(param_groups)
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["tuned_epoch"],
                eta_min=self.min_lr,
            )
        elif self.args["scheduler"] == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        elif self.args["scheduler"] == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.args['scheduler']}")
        return scheduler

    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                local_targets = targets - self._known_classes

                logits = self._network.forward_train_task(inputs, expert_idx=self._cur_task)
                loss = F.cross_entropy(logits, local_targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(local_targets).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _global_lookup_tensor(self, mapping: List[Dict[str, int]], device: torch.device) -> torch.Tensor:
        lookup = [
            self.task_offsets[item["expert_idx"]] + item["local_class_idx"]
            for item in mapping
        ]
        return torch.tensor(lookup, device=device, dtype=torch.long)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_experts = list(range(self._cur_task + 1))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                pred = self._network.predict_all(inputs, active_experts=active_experts)
                logits = pred["logits"]
                mapping = pred["mapping"]

            topk = min(self.topk, logits.shape[1])
            topk_concat = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1]
            global_lookup = self._global_lookup_tensor(mapping, device=logits.device)
            predicts = global_lookup[topk_concat]

            if topk < self.topk:
                pad = torch.full(
                    (predicts.shape[0], self.topk - topk),
                    -1,
                    device=predicts.device,
                    dtype=predicts.dtype,
                )
                predicts = torch.cat([predicts, pad], dim=1)

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
