import logging
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import TunaLinear
from backbone.vit_spie import ExpertViT
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
            f"Unsupported onlymax backbone_type '{backbone_type}'. Supported values: {sorted(valid)} "
            "with or without the '_spie' suffix."
        )
    return name


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type="cosface", eps=1e-7, s=20, m=0):
        super().__init__()
        loss_type = loss_type.lower()
        assert loss_type in ["arcface", "sphereface", "cosface", "crossentropy"]
        if loss_type == "arcface":
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == "cosface":
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m

        self.loss_type = loss_type
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == "crossentropy":
            return self.cross_entropy(wf * self.s, labels)

        if self.loss_type == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == "arcface":
            numerator = self.s * torch.cos(
                torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
                + self.m
            )
        if self.loss_type == "sphereface":
            numerator = self.s * torch.cos(
                self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
            )

        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        loss = numerator - torch.log(denominator)
        return -torch.mean(loss)


class OnlyMaxNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = ExpertViT(
            model_name=_resolve_timm_model_name(args["backbone_type"]),
            pretrained=args.get("pretrained", True),
            num_experts=len(
                _build_task_class_counts(
                    total_classes=args["nb_classes"],
                    init_cls=args["init_cls"],
                    increment=args["increment"],
                )
            ),
            expert_tokens=args.get("expert_tokens", 4),
            lora_rank=args.get("lora_rank", 8),
            lora_alpha=args.get("lora_alpha", 1.0),
            freeze_backbone=args.get("freeze_backbone", True),
        )
        self.fc = None

    @property
    def feature_dim(self):
        return self.backbone.embed_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = TunaLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=False)

    def expert_parameters_for_expert(self, expert_idx: int) -> Iterable[nn.Parameter]:
        yield from self.backbone.expert_parameters_for_expert(expert_idx)

    def parameter_groups_for_single_task(
        self,
        expert_idx: int,
        lr: float,
        head_lr: Optional[float] = None,
        expert_weight_decay: float = 0.0,
        head_weight_decay: float = 0.0,
    ) -> List[Dict[str, object]]:
        head_lr = lr if head_lr is None else head_lr
        return [
            {
                "params": list(self.backbone.expert_parameters_for_expert(expert_idx)),
                "lr": lr,
                "weight_decay": expert_weight_decay,
            },
            {
                "params": list(self.fc.parameters()),
                "lr": head_lr,
                "weight_decay": head_weight_decay,
            },
        ]

    def forward_single_expert(self, x: torch.Tensor, expert_idx: int) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, active_experts=[expert_idx], return_dict=True)
        features = out.expert_pooled[:, 0, :]
        logits = self.fc(features)["logits"]
        return {
            "features": features,
            "logits": logits,
        }

    @torch.no_grad()
    def predict_onlymax(
        self,
        x: torch.Tensor,
        active_experts: Optional[Union[Sequence[int], torch.Tensor]] = None,
        total_classes: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(x, active_experts=active_experts, return_dict=True)
        logits_per_expert = []
        for local_idx in range(out.expert_pooled.shape[1]):
            logits = self.fc(out.expert_pooled[:, local_idx, :])["logits"]
            if total_classes is not None:
                logits = logits[:, :total_classes]
            logits_per_expert.append(logits)

        stacked_logits = torch.stack(logits_per_expert, dim=0)
        max_logits, best_expert_per_class = torch.max(stacked_logits, dim=0)
        best_scores_per_expert = stacked_logits.max(dim=2).values.transpose(0, 1)
        best_expert_per_sample = best_scores_per_expert.argmax(dim=1)
        selected_logits = stacked_logits.permute(1, 0, 2)[
            torch.arange(stacked_logits.shape[1], device=stacked_logits.device),
            best_expert_per_sample,
        ]
        return {
            "logits": selected_logits,
            "selected_logits": selected_logits,
            "max_logits_per_class": max_logits,
            "stacked_logits": stacked_logits,
            "best_expert_per_class": best_expert_per_class,
            "best_expert_per_sample": best_expert_per_sample,
        }


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.ca_lr = args["ca_lr"]
        self.crct_epochs = args["crct_epochs"]
        self.args = args
        self.args["tuned_epoch"] = args["tuned_epoch"]

        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        self.task_class_counts = _build_task_class_counts(
            total_classes=args["nb_classes"],
            init_cls=args["init_cls"],
            increment=args["increment"],
        )

        self._network = OnlyMaxNet(args)

        total_params = sum(p.numel() for p in self._network.parameters())
        trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f"{total_params:,} model total parameters.")
        logging.info(f"{trainable_params:,} model trainable parameters.")

        if len(self._multiple_gpus) > 1:
            logging.info("onlymax currently uses a single device and does not wrap the model with DataParallel.")

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for class_idx in range(self._known_classes, self._total_classes):
            self.cls2task[class_idx] = self._cur_task

        self._network.update_fc(self._total_classes - self._known_classes)
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
        self._compute_mean()
        if self._cur_task > 0:
            self.classifer_align()

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
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface",
            eps=1e-7,
            s=self.args["scale"],
            m=self.args["m"],
        )
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                local_targets = targets - self._known_classes

                out = self._network.forward_single_expert(inputs, expert_idx=self._cur_task)
                logits = out["logits"][:, : self._total_classes]
                loss = loss_cos(logits[:, self._known_classes : self._total_classes], local_targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
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

    @torch.no_grad()
    def _compute_mean(self):
        self._network.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            _, _, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.batch_size * 3,
                shuffle=False,
                num_workers=4,
            )

            vectors = []
            expert_idx = self.cls2task[class_idx]
            for _, inputs, _ in idx_loader:
                features = self._network.forward_single_expert(inputs.to(self._device), expert_idx=expert_idx)["features"]
                vectors.append(features)
            vectors = torch.cat(vectors, dim=0)

            features_per_cls = vectors
            self.cls_mean[class_idx] = features_per_cls.mean(dim=0).to(self._device)
            if self.args["ca_storage_efficient_method"] == "covariance":
                self.cls_cov[class_idx] = torch.cov(features_per_cls.T) + (
                    torch.eye(self.cls_mean[class_idx].shape[-1], device=self._device) * 1e-4
                )
            elif self.args["ca_storage_efficient_method"] == "variance":
                self.cls_cov[class_idx] = torch.diag(
                    torch.cov(features_per_cls.T)
                    + (torch.eye(self.cls_mean[class_idx].shape[-1], device=self._device) * 1e-4)
                )
            else:
                raise NotImplementedError(f"Unknown CA storage method: {self.args['ca_storage_efficient_method']}")

    def classifer_align(self):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        optimizer = optim.SGD(
            [{"params": self._network.fc.parameters(), "lr": self.ca_lr, "weight_decay": self.weight_decay}],
            lr=self.ca_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max(1, self.crct_epochs))
        prog_bar = tqdm(range(self.crct_epochs))
        self._network.eval()

        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            for class_idx in range(self._total_classes):
                if self.args["decay"]:
                    task_id = self.cls2task[class_idx]
                    decay = (task_id + 1) / (self._cur_task + 1) * 0.1
                    mean = self.cls_mean[class_idx].to(self._device) * (0.9 + decay)
                else:
                    mean = self.cls_mean[class_idx].to(self._device)

                cov = self.cls_cov[class_idx].to(self._device)
                if self.args["ca_storage_efficient_method"] == "variance":
                    cov = torch.diag(cov)

                dist = MultivariateNormal(mean.float(), cov.float())
                sampled_data.append(dist.sample(sample_shape=(num_sampled_pcls,)))
                sampled_label.extend([class_idx] * num_sampled_pcls)

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label, dtype=torch.long, device=self._device)

            shuffle_idx = torch.randperm(inputs.size(0))
            inputs = inputs[shuffle_idx]
            targets = targets[shuffle_idx]

            losses = 0.0
            correct, total = 0, 0
            for iter_idx in range(self._total_classes):
                start = iter_idx * num_sampled_pcls
                end = (iter_idx + 1) * num_sampled_pcls
                inp = inputs[start:end]
                tgt = targets[start:end]
                outputs = self._network.fc(inp)["logits"][:, : self._total_classes]
                logits = self.args["scale"] * outputs

                loss = F.cross_entropy(logits, tgt)
                preds = logits.argmax(dim=1)
                correct += preds.eq(tgt).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / self._total_classes
            lr = optimizer.param_groups[0]["lr"]
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                avg_loss,
                ca_acc,
            )
            self._record_extra_stage_epoch(
                stage="classifier_align",
                epoch=epoch + 1,
                total_epochs=self.crct_epochs,
                loss=float(avg_loss),
                acc=float(ca_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_experts = list(range(self._cur_task + 1))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.predict_onlymax(
                    inputs,
                    active_experts=active_experts,
                    total_classes=self._total_classes,
                )["logits"] * self.args["scale"]

            topk = min(self.topk, outputs.shape[1])
            predicts = torch.topk(outputs, k=topk, dim=1, largest=True, sorted=True)[1]
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
