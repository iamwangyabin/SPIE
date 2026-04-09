import logging

import numpy as np
import torch
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.spie_v8 import Learner as SPiEV8Learner
from models.tuna import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy


class Learner(SPiEV8Learner):
    """SPiE v9 learner with expert calibration and a dedicated shared-CLS classifier."""

    _spie_version_name = "SPiE v9"

    def __init__(self, args):
        args["enable_shared_cls_classifier"] = True
        args["enable_expert_calibration"] = True
        super().__init__(args)

        self.expert_calibration_epochs = int(args.get("expert_calibration_epochs", 5))
        self.expert_calibration_lr = float(
            args.get("expert_calibration_lr", self.init_lr * args.get("expert_calibration_lr_scale", 0.1))
        )
        self.expert_calibration_weight_decay = float(
            args.get("expert_calibration_weight_decay", 0.0)
        )
        self.expert_calibration_routing_temperature = float(
            args.get("expert_calibration_routing_temperature", 1.0)
        )

        self.shared_cls_epochs = int(args.get("shared_cls_epochs", args["tuned_epoch"]))
        self.shared_cls_lr = float(args.get("shared_cls_lr", self.init_lr))
        self.shared_cls_weight_decay = float(args.get("shared_cls_weight_decay", self.weight_decay))
        self.shared_cls_ca_lr = float(args.get("shared_cls_ca_lr", self.ca_lr))
        self.shared_cls_crct_epochs = int(args.get("shared_cls_crct_epochs", self.crct_epochs))

        self.shared_cls_mean = dict()
        self.shared_cls_cov = dict()

        logging.info(
            "SPiE v9 expert calibration: epochs=%s, lr=%s, routing_temperature=%s.",
            self.expert_calibration_epochs,
            self.expert_calibration_lr,
            self.expert_calibration_routing_temperature,
        )
        logging.info(
            "SPiE v9 shared-CLS classifier: epochs=%s, lr=%s, crct_epochs=%s, crct_lr=%s.",
            self.shared_cls_epochs,
            self.shared_cls_lr,
            self.shared_cls_crct_epochs,
            self.shared_cls_ca_lr,
        )

    @staticmethod
    def _restore_requires_grad(params, flags):
        for param, requires_grad in zip(params, flags):
            param.requires_grad = requires_grad

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        if self._network.fc_shared_cls is not None:
            self._network.fc_shared_cls.to(self._device)

        if self._cur_task == 0:
            self._train_task0_shared_lora(train_loader)
            self._train_task0_expert(train_loader)
        else:
            self._train_incremental_expert(train_loader)
            self._train_shared_delta(train_loader)

        self._train_expert_calibration(train_loader)

        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

        self._train_shared_cls_classifier(train_loader)
        self._compute_shared_cls_mean(backbone)
        if self._cur_task > 0:
            self._classifier_align_shared_cls()

    def _active_expert_ids(self):
        return list(range(self._cur_task + 1))

    def _expert_calibration_optimizer(self):
        params = []
        for expert_id in self._active_expert_ids():
            params.append(self._network.expert_calibration_log_scale[expert_id])
            params.append(self._network.expert_calibration_bias[expert_id])
        network_params = [
            {
                "params": params,
                "lr": self.expert_calibration_lr,
                "weight_decay": self.expert_calibration_weight_decay,
            }
        ]
        return self._make_optimizer(network_params)

    def _train_expert_calibration(self, train_loader):
        if self.expert_calibration_epochs <= 0:
            return

        active_expert_ids = self._active_expert_ids()
        if not active_expert_ids:
            return

        optimizer = self._expert_calibration_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.expert_calibration_epochs)
        prog_bar = tqdm(range(self.expert_calibration_epochs))

        backbone_params = list(self._network.backbone.parameters())
        fc_params = list(self._network.fc.parameters())
        backbone_grad_flags = [param.requires_grad for param in backbone_params]
        fc_grad_flags = [param.requires_grad for param in fc_params]

        try:
            self._network.backbone.eval()
            self._network.fc.eval()
            for param in backbone_params:
                param.requires_grad = False
            for param in fc_params:
                param.requires_grad = False

            routing_temperature = max(self.expert_calibration_routing_temperature, 1e-8)
            for _, epoch in enumerate(prog_bar):
                losses = 0.0
                correct, total = 0, 0

                for _, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    frozen_logits = []
                    with torch.no_grad():
                        for expert_id in active_expert_ids:
                            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
                            logits = self._network.fc(features)["logits"][:, : self._total_classes]
                            frozen_logits.append(logits.detach())

                    stacked_logits = torch.stack(
                        [
                            self._network.calibrate_expert_logits(logits, expert_id)
                            for expert_id, logits in zip(active_expert_ids, frozen_logits)
                        ],
                        dim=0,
                    )
                    expert_scores = stacked_logits.max(dim=2).values
                    expert_weights = F.softmax(expert_scores / routing_temperature, dim=0)
                    fused_logits = (expert_weights.unsqueeze(-1) * stacked_logits).sum(dim=0)

                    loss = F.cross_entropy(self.args["scale"] * fused_logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses += loss.item()
                    _, preds = torch.max(fused_logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                if scheduler:
                    scheduler.step()

                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = losses / len(train_loader)
                info = "Task {}, expert_calibration, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.expert_calibration_epochs,
                    avg_loss,
                    train_acc,
                )
                self._record_extra_stage_epoch(
                    stage="expert_calibration",
                    epoch=epoch + 1,
                    total_epochs=self.expert_calibration_epochs,
                    loss=float(avg_loss),
                    acc=float(train_acc),
                    lr=float(lr),
                )
                prog_bar.set_description(info)
        finally:
            self._restore_requires_grad(backbone_params, backbone_grad_flags)
            self._restore_requires_grad(fc_params, fc_grad_flags)

        logging.info(info)

    def _shared_cls_optimizer(self):
        network_params = [
            {
                "params": self._network.fc_shared_cls.parameters(),
                "lr": self.shared_cls_lr,
                "weight_decay": self.shared_cls_weight_decay,
            }
        ]
        return self._make_optimizer(network_params)

    def _train_shared_cls_classifier(self, train_loader):
        if self.shared_cls_epochs <= 0 or self._network.fc_shared_cls is None:
            return

        optimizer = self._shared_cls_optimizer()
        scheduler = self._get_scheduler_for_epochs(optimizer, self.shared_cls_epochs)
        prog_bar = tqdm(range(self.shared_cls_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["m"])

        self._network.backbone.eval()
        for _, epoch in enumerate(prog_bar):
            self._network.fc_shared_cls.train()
            losses = 0.0
            correct, total = 0, 0

            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.no_grad():
                    cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
                logits = self._network.fc_shared_cls(cls_features)["logits"]

                loss = loss_cos(logits[:, self._known_classes :], targets - self._known_classes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits[:, self._known_classes :], dim=1)
                preds = preds + self._known_classes
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = losses / len(train_loader)
            info = "Task {}, shared_cls_classifier, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.shared_cls_epochs,
                avg_loss,
                train_acc,
            )
            self._record_extra_stage_epoch(
                stage="shared_cls_classifier",
                epoch=epoch + 1,
                total_epochs=self.shared_cls_epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _stack_calibrated_expert_logits(self, inputs, active_expert_ids):
        logits_per_expert = []
        for expert_id in active_expert_ids:
            features = self._network.backbone(inputs, adapter_id=expert_id, train=False)["features"]
            logits = self._network.fc(features)["logits"][:, : self._total_classes]
            logits_per_expert.append(self._network.calibrate_expert_logits(logits, expert_id))
        return torch.stack(logits_per_expert, dim=0)

    def _fuse_calibrated_expert_logits(self, stacked_logits):
        routing_temperature = max(self.expert_calibration_routing_temperature, 1e-8)
        expert_scores = stacked_logits.max(dim=2).values
        expert_weights = F.softmax(expert_scores / routing_temperature, dim=0)
        return (expert_weights.unsqueeze(-1) * stacked_logits).sum(dim=0)

    def _shared_cls_logits(self, inputs):
        if self._network.fc_shared_cls is None:
            return None
        cls_features = self._network.backbone(inputs, adapter_id=-1, train=False)["cls_features"]
        return self._network.fc_shared_cls(cls_features)["logits"][:, : self._total_classes]

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_expert_ids = self._active_expert_ids()
        use_shared_cls = bool(self.args.get("v9_use_shared_cls_eval", True))
        expert_weight = float(self.args.get("v9_expert_fusion_weight", 1.0))
        shared_weight = float(self.args.get("v9_shared_cls_fusion_weight", 1.0))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                stacked_logits = self._stack_calibrated_expert_logits(inputs, active_expert_ids)
                logits = self._fuse_calibrated_expert_logits(stacked_logits)

                if use_shared_cls:
                    shared_cls_logits = self._shared_cls_logits(inputs)
                    if shared_cls_logits is not None:
                        logits = expert_weight * logits + shared_weight * shared_cls_logits

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

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    @torch.no_grad()
    def _compute_shared_cls_mean(self, model):
        model.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size * 3, shuffle=False, num_workers=4
            )

            vectors = []
            for _, _inputs, _targets in idx_loader:
                res = model(_inputs.to(self._device), adapter_id=-1, train=False)
                vectors.append(res["cls_features"])
            vectors = torch.cat(vectors, dim=0)

            if self.args["ca_storage_efficient_method"] == "covariance":
                self.shared_cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
                self.shared_cls_cov[class_idx] = torch.cov(vectors.T) + (
                    torch.eye(self.shared_cls_mean[class_idx].shape[-1]) * 1e-4
                ).to(self._device)
            elif self.args["ca_storage_efficient_method"] == "variance":
                self.shared_cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)
                self.shared_cls_cov[class_idx] = torch.diag(
                    torch.cov(vectors.T)
                    + (torch.eye(self.shared_cls_mean[class_idx].shape[-1]) * 1e-4).to(self._device)
                )

    def _classifier_align_module(self, classifier, mean_dict, cov_dict, stage, run_epochs, lr):
        if classifier is None:
            return

        for p in classifier.parameters():
            p.requires_grad = True

        network_params = [{"params": classifier.parameters(), "lr": lr, "weight_decay": self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5)

        prog_bar = tqdm(range(run_epochs))
        task_size = self._known_classes - self._total_classes
        self._network.eval()
        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            if self.args["ca_storage_efficient_method"] in ["covariance", "variance"]:
                for class_idx in range(self._total_classes):
                    if self.args["decay"]:
                        t_id = class_idx // task_size
                        decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                        mean = torch.tensor(mean_dict[class_idx], dtype=torch.float64).to(self._device) * (0.9 + decay)
                    else:
                        mean = mean_dict[class_idx].to(self._device)
                    cov = cov_dict[class_idx].to(self._device)
                    if self.args["ca_storage_efficient_method"] == "variance":
                        cov = torch.diag(cov)
                    distribution = MultivariateNormal(mean.float(), cov.float())
                    sampled_data.append(distribution.sample(sample_shape=(num_sampled_pcls,)))
                    sampled_label.extend([class_idx] * num_sampled_pcls)
            else:
                raise NotImplementedError

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            shuffle_indices = torch.randperm(inputs.size(0))
            inputs = inputs[shuffle_indices]
            targets = targets[shuffle_indices]

            losses = 0.0
            correct, total = 0, 0
            for _iter in range(self._total_classes):
                inp = inputs[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]
                outputs = classifier(inp)["logits"]
                logits = self.args["scale"] * outputs

                loss = F.cross_entropy(logits, tgt)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(tgt.expand_as(preds)).cpu().sum()
                total += len(tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / self._total_classes
            lr_value = optimizer.param_groups[0]["lr"]
            info = "Task {}, {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                stage,
                epoch + 1,
                run_epochs,
                avg_loss,
                ca_acc,
            )
            self._record_extra_stage_epoch(
                stage=stage,
                epoch=epoch + 1,
                total_epochs=run_epochs,
                loss=float(avg_loss),
                acc=float(ca_acc),
                lr=float(lr_value),
            )
            prog_bar.set_description(info)

        logging.info(info)

    def _classifier_align_shared_cls(self):
        self._classifier_align_module(
            classifier=self._network.fc_shared_cls,
            mean_dict=self.shared_cls_mean,
            cov_dict=self.shared_cls_cov,
            stage="shared_cls_classifier_align",
            run_epochs=self.shared_cls_crct_epochs,
            lr=self.shared_cls_ca_lr,
        )
