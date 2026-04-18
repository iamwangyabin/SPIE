import logging

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.linears import TunaLinear
from models.base import BaseLearner
from models.tuna import AngularPenaltySMLoss
from utils.inc_net import get_backbone
from utils.toolkit import tensor2numpy

num_workers = 8


class SPIEV17Net(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        self.backbone = get_backbone(args, pretrained)
        self.backbone.out_dim = getattr(self.backbone, "out_dim", 768)
        self.fc_shared_cls = None
        self.expert_heads = nn.ModuleList()
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    @property
    def expert_feature_dim(self):
        return self.backbone.out_dim * 2

    def generate_fc(self, in_dim, out_dim):
        return TunaLinear(in_dim, out_dim)

    def update_fc(self, nb_classes):
        if self.fc_shared_cls is None:
            self.fc_shared_cls = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc_shared_cls.update(nb_classes, freeze_old=False)

    def append_expert_head(self, nb_classes):
        for head in self.expert_heads:
            head.requires_grad_(False)
        head = self.generate_fc(self.expert_feature_dim, nb_classes).to(self._device)
        self.expert_heads.append(head)
        return head

    def get_expert_head(self, task_id):
        if task_id >= len(self.expert_heads):
            raise IndexError(f"Expert head {task_id} is not initialized.")
        return self.expert_heads[task_id]

    def forward_shared_cls(self, x, train=False):
        if self.fc_shared_cls is None:
            raise RuntimeError("fc_shared_cls is not initialized.")
        res = self.backbone(x, adapter_id=-1, train=train)
        cls_features = res["cls_features"]
        return {
            "cls_features": cls_features,
            "logits": self.fc_shared_cls(cls_features)["logits"],
        }

    def forward(self, x, adapter_id=-1, train=False, fc_only=False):
        return self.backbone(x, adapter_id, train, fc_only)


class Learner(BaseLearner):
    """SPiE v17 clean ablation with frozen shared LoRA and trainable expert adapters."""

    _spie_version_name = "SPiE v17"

    def __init__(self, args):
        super().__init__(args)

        self._network = SPIEV17Net(args, True)
        self.task_class_ranges = []

        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.min_lr = args["min_lr"]
        self.args = args
        self.share_lora_weight_decay = float(args["share_lora_weight_decay"])
        self.expert_head_weight_decay = float(args["expert_head_weight_decay"])

        self.task0_shared_epochs = int(args["task0_shared_epochs"])
        self.task0_shared_lr = float(args["task0_shared_lr"])
        self.shared_cls_epochs = int(args["shared_cls_epochs"])
        self.shared_cls_lr = float(args["shared_cls_lr"])
        self.shared_cls_weight_decay = float(args["shared_cls_weight_decay"])
        self.freeze_shared_lora_after_task0 = bool(args.get("freeze_shared_lora_after_task0", True))

        self.task0_expert_epochs = int(args["task0_expert_epochs"])
        self.task0_expert_lr = float(args["task0_expert_lr"])
        self.incremental_expert_epochs = int(args["incremental_expert_epochs"])
        self.incremental_expert_lr = float(args["incremental_expert_lr"])
        self.relation_alpha = float(args.get("relation_alpha", 0.2))

        for name, param in self._network.backbone.named_parameters():
            param.requires_grad = (
                "cur_adapter" in name
                or "cur_expert_tokens" in name
                or "cur_shared_adapter" in name
            )

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s %s total backbone parameters.", f"{total_params:,}", self._spie_version_name)
        logging.info("%s %s trainable backbone parameters.", f"{total_trainable_params:,}", self._spie_version_name)
        logging.info(
            (
                "SPiE v17 clean shared branch: task0 epochs=%s lr=%s, incremental head epochs=%s lr=%s, "
                "freeze_shared_lora_after_task0=%s."
            ),
            self.task0_shared_epochs,
            self.task0_shared_lr,
            self.shared_cls_epochs,
            self.shared_cls_lr,
            self.freeze_shared_lora_after_task0,
        )
        logging.info(
            "SPiE v17 clean expert branch: task0 epochs=%s lr=%s, incremental epochs=%s lr=%s, relation_alpha=%s.",
            self.task0_expert_epochs,
            self.task0_expert_lr,
            self.incremental_expert_epochs,
            self.incremental_expert_lr,
            self.relation_alpha,
        )

    def _backbone_module(self):
        if isinstance(self._network.backbone, nn.DataParallel):
            return self._network.backbone.module
        return self._network.backbone

    def after_task(self):
        self._known_classes = self._total_classes

    def _should_reset_task_modules(self):
        return self._cur_task >= 0

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        current_task_size = self._total_classes - self._known_classes
        self.task_class_ranges.append((self._known_classes, self._total_classes))

        self._network.update_fc(current_task_size)
        self._network.append_expert_head(current_task_size)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        if self._should_reset_task_modules():
            self._backbone_module().reset_task_modules()

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
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        use_backbone_dataparallel = bool(self.args.get("spie_v15_backbone_dataparallel", False))
        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = nn.DataParallel(self._network.backbone, self._multiple_gpus)

        self._train(self.train_loader)

        if use_backbone_dataparallel and len(self._multiple_gpus) > 1:
            self._network.backbone = self._backbone_module()

    def _make_optimizer(self, network_params):
        if self.args["optimizer"] == "sgd":
            return optim.SGD(network_params, momentum=0.9)
        if self.args["optimizer"] == "adam":
            return optim.Adam(network_params)
        if self.args["optimizer"] == "adamw":
            return optim.AdamW(network_params)
        raise ValueError(f"Unsupported optimizer: {self.args['optimizer']}")

    def _get_scheduler_for_epochs(self, optimizer, epochs):
        if self.args["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=self.min_lr)
        if self.args["scheduler"] == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        if self.args["scheduler"] == "constant":
            return None
        raise ValueError(f"Unsupported scheduler: {self.args['scheduler']}")

    def _set_shared_lora_requires_grad(self, requires_grad):
        self._backbone_module().cur_shared_adapter.requires_grad_(requires_grad)

    def _set_current_expert_requires_grad(self, requires_grad):
        backbone = self._backbone_module()
        backbone.cur_adapter.requires_grad_(requires_grad)
        backbone.cur_expert_tokens.requires_grad = requires_grad

    def _set_expert_head_requires_grad(self, task_id, requires_grad):
        self._network.get_expert_head(task_id).requires_grad_(requires_grad)

    def _shared_branch_optimizer(self, lr):
        backbone = self._backbone_module()
        network_params = []
        shared_lora_params = [p for p in backbone.cur_shared_adapter.parameters() if p.requires_grad]
        if shared_lora_params:
            network_params.append(
                {
                    "params": shared_lora_params,
                    "lr": lr,
                    "weight_decay": self.share_lora_weight_decay,
                }
            )
        network_params.append(
            {
                "params": self._network.fc_shared_cls.parameters(),
                "lr": self.shared_cls_lr,
                "weight_decay": self.shared_cls_weight_decay,
            }
        )
        return self._make_optimizer(network_params)

    def _current_expert_optimizer(self, lr):
        backbone = self._backbone_module()
        expert_params = [
            p
            for name, p in backbone.named_parameters()
            if p.requires_grad and ("cur_adapter" in name or "cur_expert_tokens" in name)
        ]
        expert_head_params = [p for p in self._network.get_expert_head(self._cur_task).parameters() if p.requires_grad]
        network_params = [
            {
                "params": expert_params,
                "lr": lr,
                "weight_decay": self.weight_decay,
            },
            {
                "params": expert_head_params,
                "lr": lr,
                "weight_decay": self.expert_head_weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    def _collect_expert_logits(self, inputs, task_ids):
        if not task_ids:
            return {}

        task_ids = list(task_ids)
        backbone = self._backbone_module()
        if len(task_ids) > 1 and hasattr(backbone, "forward_multi_expert_features"):
            res = backbone.forward_multi_expert_features(inputs, task_ids)
            expert_feature_map = {
                task_id: res["expert_features"][local_idx]
                for local_idx, task_id in enumerate(task_ids)
            }
        else:
            expert_feature_map = {
                task_id: self._network.backbone(inputs, adapter_id=task_id, train=False)["expert_features"]
                for task_id in task_ids
            }

        return {
            task_id: self._network.get_expert_head(task_id)(expert_feature_map[task_id])["logits"]
            for task_id in task_ids
        }

    def _train(self, train_loader):
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc_shared_cls.to(self._device)
        self._network.expert_heads.to(self._device)

        if self._cur_task == 0:
            self._train_shared_branch(
                train_loader=train_loader,
                epochs=self.task0_shared_epochs,
                branch_lr=self.task0_shared_lr,
                stage="task0_shared_branch",
            )
            self._train_current_expert(
                train_loader=train_loader,
                epochs=self.task0_expert_epochs,
                expert_lr=self.task0_expert_lr,
                stage="task0_expert_local",
            )
        else:
            self._train_shared_branch(
                train_loader=train_loader,
                epochs=self.shared_cls_epochs,
                branch_lr=self.shared_cls_lr,
                stage="shared_cls_head_only",
            )
            self._train_current_expert(
                train_loader=train_loader,
                epochs=self.incremental_expert_epochs,
                expert_lr=self.incremental_expert_lr,
                stage="incremental_expert_local",
            )

        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(False)
        self._set_expert_head_requires_grad(self._cur_task, False)
        backbone_module.adapter_update()

    def _train_shared_branch(self, train_loader, epochs, branch_lr, stage):
        if epochs <= 0 or self._network.fc_shared_cls is None:
            return

        train_shared_lora = self._cur_task == 0 and not self.freeze_shared_lora_after_task0
        if self._cur_task == 0:
            train_shared_lora = True
        self._set_shared_lora_requires_grad(train_shared_lora)
        self._set_current_expert_requires_grad(False)
        self._set_expert_head_requires_grad(self._cur_task, False)

        optimizer = self._shared_branch_optimizer(branch_lr)
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            self._network.fc_shared_cls.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                cls_features = self._network.backbone(inputs, adapter_id=-1, train=True)["cls_features"]
                logits = self._network.fc_shared_cls(cls_features)["logits"]

                loss = loss_cos(logits[:, self._known_classes : self._total_classes], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits[:, self._known_classes : self._total_classes], dim=1)
                preds = preds + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                stage,
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
            )
            self._record_extra_stage_epoch(
                stage=stage,
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                shared_lora_trainable=bool(train_shared_lora),
            )
            prog_bar.set_description(info)

        logging.info(info)
        self._set_shared_lora_requires_grad(False)

    def _train_current_expert(self, train_loader, epochs, expert_lr, stage):
        if epochs <= 0:
            return

        self._set_shared_lora_requires_grad(False)
        self._set_current_expert_requires_grad(True)
        self._set_expert_head_requires_grad(self._cur_task, True)

        optimizer = self._current_expert_optimizer(expert_lr)
        scheduler = self._get_scheduler_for_epochs(optimizer, epochs)
        prog_bar = tqdm(range(epochs))
        expert_head = self._network.get_expert_head(self._cur_task)
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            expert_head.train()
            losses = 0.0
            ce_losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes

                expert_features = self._network.backbone(inputs, adapter_id=self._cur_task, train=True)["expert_features"]
                logits = expert_head(expert_features)["logits"]
                ce_loss = loss_cos(logits, local_targets)
                loss = ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                ce_losses += ce_loss.item()
                preds = torch.argmax(logits, dim=1) + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            avg_ce_loss = ce_losses / len(train_loader)
            info = "Task {}, {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                stage,
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                stage=stage,
                ce_loss=float(avg_ce_loss),
            )
            prog_bar.set_description(info)

        logging.info(info)
        self._set_current_expert_requires_grad(False)
        self._set_expert_head_requires_grad(self._cur_task, False)

    def _shared_cls_logits(self, inputs):
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        return self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]

    def _predict_topk(self, logits):
        topk = min(self.topk, logits.shape[1])
        predicts = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1]
        if topk < self.topk:
            pad = torch.full(
                (predicts.shape[0], self.topk - topk),
                -1,
                device=predicts.device,
                dtype=predicts.dtype,
            )
            predicts = torch.cat([predicts, pad], dim=1)
        return predicts

    def _centered_cosine_batch(self, lhs, rhs, eps=1e-12):
        lhs_centered = lhs - lhs.mean(dim=1, keepdim=True)
        rhs_centered = rhs - rhs.mean(dim=1, keepdim=True)
        numerator = torch.sum(lhs_centered * rhs_centered, dim=1)
        denominator = lhs_centered.norm(dim=1) * rhs_centered.norm(dim=1)
        return numerator / denominator.clamp_min(eps)

    def _relation_fusion_logits(self, shared_logits, expert_logits_by_task):
        if not self.task_class_ranges:
            return shared_logits

        fused_logits = shared_logits.clone()
        for task_id, (start_idx, end_idx) in enumerate(self.task_class_ranges):
            shared_slice = shared_logits[:, start_idx:end_idx]
            expert_slice = expert_logits_by_task[task_id]
            relation_score = self._centered_cosine_batch(shared_slice, expert_slice)
            fused_logits[:, start_idx:end_idx] += self.relation_alpha * relation_score.unsqueeze(1)
        return fused_logits

    def eval_task(self):
        cnn_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(cnn_pred, y_true)

        nme_pred, nme_true = self._eval_nme(self.test_loader, class_means=None)
        nme_accy = self._evaluate(nme_pred, nme_true)

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                logits = self._shared_cls_logits(inputs)
                predicts = self._predict_topk(logits)

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        del class_means

        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                shared_out = self._network.forward_shared_cls(inputs, train=False)
                shared_logits = shared_out["logits"][:, : self._total_classes]
                expert_logits_by_task = self._collect_expert_logits(inputs, list(range(len(self.task_class_ranges))))
                fused_logits = self._relation_fusion_logits(shared_logits, expert_logits_by_task)
                predicts = self._predict_topk(fused_logits)

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
