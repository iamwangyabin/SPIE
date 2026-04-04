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
    # 根据初始类别数和后续增量步长，预先算出每个 task 对应多少类。
    # 例如 total=100, init=10, increment=10 -> [10, 10, ..., 10]
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
    # 配置文件里允许写 *_spie 后缀，这里统一还原成 timm 原始模型名。
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
        # task_offsets[i] 表示第 i 个 expert 的全局类别起始下标。
        # 后面评估时，需要把 "expert 内部类别 id" 映射回 "全局类别 id"。
        self.task_offsets = np.cumsum([0] + self.task_class_counts[:-1]).tolist()

        # SPiE 的主体模型：
        # 1. 一个共享 ViT 主干
        # 2. 每个 task / expert 一组 expert token + expert-only LoRA
        # 3. 每个 expert 一个自己的分类头
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
        # 每次只训练当前 task 对应的分类头，其它 head 冻结。
        for i, head in enumerate(self._network.heads.heads):
            head.requires_grad_(i == expert_idx)

    def incremental_train(self, data_manager):
        # 进入下一个增量任务，并更新当前已知类别范围。
        self._reset_task_logging()
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
        # SPiE 不做额外的 task 后处理，直接完成当前 expert 的训练。
        self._network.to(self._device)
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        self._init_train(train_loader, optimizer, scheduler)

    def get_optimizer(self):
        # 只给当前 task 暴露对应的参数组：
        # 1. expert 参数（expert token / expert LoRA）
        # 2. 当前 task 的分类头
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
                # 当前 task 的类别标签转成局部标签。
                # 例如全局类 [20, 21, 22] -> 局部类 [0, 1, 2]
                local_targets = targets - self._known_classes

                # 训练时只激活一个 expert，因此输出 logits 也是当前 task 的局部类别空间。
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
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                avg_loss,
                train_acc,
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.args["tuned_epoch"],
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _global_lookup_tensor(self, mapping: List[Dict[str, int]], device: torch.device) -> torch.Tensor:
        # 把 concat 后的类别位置映射回全局类别 id。
        # mapping 中每一项都形如 {"expert_idx": i, "local_class_idx": j}
        lookup = [
            self.task_offsets[item["expert_idx"]] + item["local_class_idx"]
            for item in mapping
        ]
        return torch.tensor(lookup, device=device, dtype=torch.long)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        # 测试时激活到当前 task 为止的所有 expert。
        active_experts = list(range(self._cur_task + 1))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # predict_all 会把所有 active experts 的 logits 拼接起来返回。
                pred = self._network.predict_all(inputs, active_experts=active_experts)
                logits = pred["logits"]
                mapping = pred["mapping"]

            topk = min(self.topk, logits.shape[1])
            topk_concat = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1]
            global_lookup = self._global_lookup_tensor(mapping, device=logits.device)
            # 先在拼接空间做 top-k，再映射回真实的全局类别 id。
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
