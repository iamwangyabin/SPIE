import logging

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import MOSNet
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = MOSNet(args, True)
        self.cls_mean = {}
        self.cls_cov = {}
        self.cls2task = {}
        self.batch_size = int(args["batch_size"])
        self.init_lr = float(args["init_lr"])
        self.ca_lr = float(args["ca_lr"])
        self.crct_epochs = int(args["crct_epochs"])
        self.weight_decay = float(args["weight_decay"] if args["weight_decay"] is not None else 0.0005)
        self.min_lr = float(args["min_lr"] if args["min_lr"] is not None else 1e-8)
        self.ensemble = bool(args.get("ensemble", False))
        self.args = args

        for name, param in self._network.backbone.named_parameters():
            if "adapter" not in name and "head" not in name:
                param.requires_grad = False

        total_params = sum(param.numel() for param in self._network.backbone.parameters())
        total_trainable_params = sum(
            param.numel() for param in self._network.backbone.parameters() if param.requires_grad
        )
        logging.info("%s model total parameters.", f"{total_params:,}")
        logging.info("%s model training parameters.", f"{total_trainable_params:,}")

        if total_params != total_trainable_params:
            for name, param in self._network.backbone.named_parameters():
                if param.requires_grad:
                    logging.info("%s: %s", name, param.numel())

    def _backbone_module(self):
        if isinstance(self._network.backbone, nn.DataParallel):
            return self._network.backbone.module
        return self._network.backbone

    def _build_safe_distribution(self, mean, covariance):
        mean = mean.to(self._device, dtype=torch.float32)
        if covariance.ndim == 1:
            covariance = torch.diag(covariance)

        covariance = covariance.to(self._device, dtype=torch.float64)
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.T)

        base_jitter = float(self.args.get("covariance_regularization", 1e-4))
        max_retry_power = int(self.args.get("max_covariance_retry_power", 6))
        eye = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)

        for power in range(max_retry_power + 1):
            jitter = base_jitter * (10 ** power)
            repaired_covariance = covariance + eye * jitter
            _, info = torch.linalg.cholesky_ex(repaired_covariance)
            if torch.all(info == 0):
                scale_tril = torch.linalg.cholesky(repaired_covariance).to(dtype=torch.float32)
                return MultivariateNormal(mean, scale_tril=scale_tril)

        min_eigenvalue = torch.linalg.eigvalsh(covariance).min().item()
        jitter = max(base_jitter, -min_eigenvalue + base_jitter)
        repaired_covariance = covariance + eye * jitter
        scale_tril = torch.linalg.cholesky(repaired_covariance).to(dtype=torch.float32)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def replace_fc(self):
        model = self._network.to(self._device)
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for _, (_, data, label) in enumerate(self.train_loader_for_protonet):
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.forward_orig(data)["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_list = np.unique(self.train_dataset.labels)

        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for class_id in range(self._known_classes, self._total_classes):
            self.cls2task[class_id] = self._cur_task

        self._network.update_fc(self._total_classes)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

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

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network.backbone = nn.DataParallel(self._network.backbone, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)
        self.replace_fc()

        if len(self._multiple_gpus) > 1:
            self._network.backbone = self._backbone_module()

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        optimizer = self.get_optimizer(backbone)
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, optimizer, scheduler)
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

    def get_optimizer(self, model):
        adapter_params = [p for name, p in model.named_parameters() if "adapter" in name and p.requires_grad]
        other_trainable_params = [p for name, p in model.named_parameters() if "adapter" not in name and p.requires_grad]
        network_params = [
            {"params": adapter_params, "lr": self.init_lr, "weight_decay": self.weight_decay},
            {"params": other_trainable_params, "lr": self.init_lr * 0.1, "weight_decay": self.weight_decay},
        ]

        if self.args["optimizer"] == "sgd":
            return optim.SGD(network_params, momentum=0.9)
        if self.args["optimizer"] == "adam":
            return optim.Adam(network_params)
        if self.args["optimizer"] == "adamw":
            return optim.AdamW(network_params)
        raise ValueError(f"Unsupported optimizer {self.args['optimizer']}")

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args["tuned_epoch"],
                eta_min=self.min_lr,
            )
        if self.args["scheduler"] == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
        if self.args["scheduler"] == "constant":
            return None
        raise ValueError(f"Unsupported scheduler {self.args['scheduler']}")

    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        backbone_module = self._backbone_module()

        for epoch in prog_bar:
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, : self._total_classes]
                logits[:, : self._known_classes] = float("-inf")

                loss = F.cross_entropy(logits, targets.long())
                loss = loss + self.orth_loss(output["pre_logits"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if float(self.args.get("adapter_momentum", 0.0) or 0.0) > 0:
                    backbone_module.adapter_merge()

                losses += loss.item()
                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / len(train_loader)
            lr = optimizer.param_groups[0]["lr"]
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.args["tuned_epoch"],
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                avg_loss,
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    @torch.no_grad()
    def _compute_mean(self, model):
        model.eval()

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
            for _, inputs, _targets in idx_loader:
                outputs = model(inputs.to(self._device), adapter_id=self._cur_task, train=True)
                vectors.append(outputs["features"])

            vectors = torch.cat(vectors, dim=0)
            storage_method = self.args["ca_storage_efficient_method"]
            self.cls_mean[class_idx] = vectors.mean(dim=0).to(self._device)

            if storage_method == "covariance":
                self.cls_cov[class_idx] = torch.cov(vectors.T) + (
                    torch.eye(self.cls_mean[class_idx].shape[-1], device=self._device) * 1e-4
                )
            elif storage_method == "variance":
                covariance = torch.cov(vectors.T) + (
                    torch.eye(self.cls_mean[class_idx].shape[-1], device=self._device) * 1e-4
                )
                self.cls_cov[class_idx] = torch.diag(covariance)
            else:
                raise NotImplementedError(f"Unsupported storage method {storage_method}")

    def classifer_align(self, model):
        model.train()
        param_list = [p for name, p in model.named_parameters() if p.requires_grad and "adapter" not in name]
        optimizer = optim.SGD(
            [{"params": param_list, "lr": self.ca_lr, "weight_decay": self.weight_decay}],
            lr=self.ca_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.crct_epochs)
        prog_bar = tqdm(range(self.crct_epochs))

        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = self.batch_size * 5

            for class_idx in range(self._total_classes):
                distribution = self._build_safe_distribution(
                    self.cls_mean[class_idx],
                    self.cls_cov[class_idx],
                )
                sampled_data.append(distribution.sample(sample_shape=(num_sampled_pcls,)))
                sampled_label.extend([class_idx] * num_sampled_pcls)

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label, dtype=torch.long, device=self._device)
            shuffle_indexes = torch.randperm(inputs.size(0), device=self._device)
            inputs = inputs[shuffle_indexes]
            targets = targets[shuffle_indexes]

            losses = 0.0
            correct, total = 0, 0
            for class_idx in range(self._total_classes):
                start = class_idx * num_sampled_pcls
                end = (class_idx + 1) * num_sampled_pcls
                batch_inputs = inputs[start:end]
                batch_targets = targets[start:end]
                logits = model(batch_inputs, fc_only=True)["logits"][:, : self._total_classes]
                loss = F.cross_entropy(logits, batch_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq(batch_targets.expand_as(preds)).cpu().sum()
                total += len(batch_targets)

            scheduler.step()
            ca_acc = np.round(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / self._total_classes
            self._record_extra_stage_epoch(
                stage="classifier_align",
                epoch=epoch + 1,
                total_epochs=self.crct_epochs,
                loss=float(avg_loss),
                acc=float(ca_acc),
                lr=float(optimizer.param_groups[0]["lr"]),
            )
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, CA_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.crct_epochs,
                avg_loss,
                ca_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def orth_loss(self, features):
        if self.cls_mean:
            sample_mean = torch.stack(list(self.cls_mean.values()), dim=0).to(self._device, non_blocking=True)
            matrix = torch.cat([sample_mean, features], dim=0)
        else:
            matrix = features

        similarity = torch.matmul(matrix, matrix.t()) / 0.8
        labels = torch.arange(0, similarity.shape[0], device=self._device).long()
        loss = torch.nn.functional.cross_entropy(similarity, labels)
        return self.args["reg"] * loss

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        orig_y_pred = []
        backbone = self._backbone_module()

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                orig_logits = self._network.forward_orig(inputs)["logits"][:, : self._total_classes]
                orig_preds = torch.max(orig_logits, dim=1)[1].cpu().numpy()
                orig_idx = torch.tensor([self.cls2task[pred] for pred in orig_preds], device=self._device)
                orig_y_pred.append(orig_preds)

                all_features = torch.zeros(
                    len(inputs),
                    self._cur_task + 1,
                    backbone.out_dim,
                    device=self._device,
                )
                for task_id in range(self._cur_task + 1):
                    outputs = backbone(inputs, adapter_id=task_id, train=False)
                    all_features[:, task_id, :] = outputs["features"]

                final_logits = []
                max_iter = 4
                for sample_id in range(len(inputs)):
                    loop_num = 0
                    prev_adapter_idx = orig_idx[sample_id]
                    while True:
                        loop_num += 1
                        cur_feature = all_features[sample_id, prev_adapter_idx].unsqueeze(0)
                        cur_logits = backbone(cur_feature, fc_only=True)["logits"][:, : self._total_classes]
                        cur_pred = torch.max(cur_logits, dim=1)[1].cpu().numpy()
                        cur_adapter_idx = torch.tensor([self.cls2task[pred] for pred in cur_pred], device=self._device)[0]
                        if loop_num >= max_iter or cur_adapter_idx == prev_adapter_idx:
                            break
                        prev_adapter_idx = cur_adapter_idx
                    final_logits.append(cur_logits)

                final_logits = torch.cat(final_logits, dim=0).to(self._device)
                if self.ensemble:
                    final_logits = F.softmax(final_logits, dim=1)
                    orig_logits = F.softmax(orig_logits / (1 / (self._cur_task + 1)), dim=1)
                    outputs = final_logits + orig_logits
                else:
                    outputs = final_logits

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        orig_acc = (np.concatenate(orig_y_pred) == np.concatenate(y_true)).sum() * 100 / len(np.concatenate(y_true))
        logging.info("the accuracy of the original model:%.2f", np.around(orig_acc, 2))
        return np.concatenate(y_pred), np.concatenate(y_true)
