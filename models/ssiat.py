import logging
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone import vit_ssiat_adapter
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

FEATURE_STATS_BATCH_SIZE = 64
FEATURE_STATS_WORKERS = 4


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type="cosface", eps=1e-7, s=20.0, m=0.0):
        super().__init__()
        loss_type = loss_type.lower()
        assert loss_type in {"arcface", "sphereface", "cosface", "crossentropy"}
        self.loss_type = loss_type
        self.eps = eps
        self.s = s
        self.m = m
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == "crossentropy":
            return self.cross_entropy(wf, labels)

        diag = torch.diagonal(wf.transpose(0, 1)[labels])
        if self.loss_type == "cosface":
            numerator = self.s * (diag - self.m)
        elif self.loss_type == "arcface":
            numerator = self.s * torch.cos(torch.acos(torch.clamp(diag, -1.0 + self.eps, 1 - self.eps)) + self.m)
        else:
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(diag, -1.0 + self.eps, 1 - self.eps)))

        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        loss = numerator - torch.log(denominator)
        return -torch.mean(loss)


class SSIATContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm

        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))
        single_head.append(nn.Linear(embed_dim, nb_classes, bias=False))

        self.heads = nn.ModuleList([nn.Sequential(*single_head)])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
        linear = nn.Linear(self.embed_dim, nb_classes, bias=False)
        trunc_normal_(linear.weight, std=0.02)
        single_head.append(linear)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for param in self.heads.parameters():
                param.requires_grad = False

        self.heads.append(new_head)

    def forward(self, x):
        logits = []
        for task_id in range(len(self.heads)):
            fc_input = x[task_id] if self.feat_expand else x
            linear = self.heads[task_id][-1]
            logits.append(
                F.linear(
                    F.normalize(fc_input, p=2, dim=1),
                    F.normalize(linear.weight, p=2, dim=1),
                )
            )
        return {"logits": torch.cat(logits, dim=1)}


def build_ssiat_backbone(args):
    backbone_type = args.get("backbone_type", args.get("convnet_type", "")).lower()
    tuning_config = SimpleNamespace(
        ffn_adapt=True,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args["ffn_num"],
        d_model=768,
        vpt_on=False,
        vpt_num=0,
    )

    if backbone_type == "pretrained_vit_b16_224_adapter":
        model = vit_ssiat_adapter.vit_base_patch16_224_adapter(
            num_classes=0,
            global_pool=False,
            drop_path_rate=0.0,
            tuning_config=tuning_config,
        )
    elif backbone_type == "pretrained_vit_b16_224_in21k_adapter":
        model = vit_ssiat_adapter.vit_base_patch16_224_in21k_adapter(
            num_classes=0,
            global_pool=False,
            drop_path_rate=0.0,
            tuning_config=tuning_config,
        )
    else:
        raise NotImplementedError(f"Unknown SSIAT backbone type: {backbone_type}")

    model.out_dim = 768
    return model.eval()


class SSIATNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super().__init__()
        self.convnet = build_ssiat_backbone(args)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def update_fc(self, nb_classes, freeze_old=False):
        if self.fc is None:
            self.fc = self.generate_fc(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=freeze_old)

    def generate_fc(self, in_dim, out_dim):
        return SSIATContinualLinear(in_dim, out_dim)

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x, fc_only=False):
        if fc_only:
            return self.fc(x)
        features = self.convnet(x)
        out = self.fc(features)
        out.update({"features": features})
        return out

    def ca_forward(self, x):
        return self.fc(x)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        backbone_type = args.get("backbone_type", args.get("convnet_type", ""))
        if "adapter" not in backbone_type:
            raise NotImplementedError("SSIAT requires an adapter backbone")

        self._network = SSIATNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args.get("weight_decay") is not None else 0.0005
        self.min_lr = args["min_lr"] if args.get("min_lr") is not None else 1e-8
        self.init_epochs = args.get("init_epochs", args.get("tuned_epoch", 20))
        self.inc_epochs = args.get("inc_epochs", args.get("tuned_epoch", 10))
        self.ca_epochs = args.get("ca_epochs", 5)
        self.num_workers = args.get("num_workers", 8)
        self.eval_workers = args.get("eval_workers", self.num_workers)
        self.scale = args.get("scale", 20.0)
        self.margin = args.get("margin", args.get("m", 0.0))
        self.logit_norm = args.get("ca_with_logit_norm")
        self.tuned_epochs = None
        self.task_sizes = []

    def after_task(self):
        self._known_classes = self._total_classes

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        return self._evaluate(y_pred, y_true), None

    def _evaluate(self, y_pred, y_true):
        grouped = self._official_accuracy(
            y_pred.T[0],
            y_true,
            self._known_classes,
            self.args["increment"],
        )
        return {
            "grouped": grouped,
            "top1": grouped["total"],
            f"top{self.topk}": np.around(
                (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
                decimals=2,
            ).item(),
        }

    @staticmethod
    def _official_accuracy(y_pred, y_true, nb_old, increment):
        assert len(y_pred) == len(y_true), "Data length error."
        all_acc = {}
        all_acc["total"] = np.around(
            (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
        )

        for class_id in range(0, np.max(y_true), increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )
            all_acc[label] = np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )

        idxes = np.where(y_true < nb_old)[0]
        all_acc["old"] = (
            0
            if len(idxes) == 0
            else np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in all_acc.items()
        }

    def extract_features(self, loader, model):
        model.eval()
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for _, data, label in loader:
                data = data.to(self._device)
                if isinstance(model, nn.DataParallel):
                    embedding = model.module.extract_vector(data)
                else:
                    embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0).numpy()
        label_list = torch.cat(label_list, dim=0).numpy()
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        self.topk = min(self.topk, self._total_classes)
        self._network.update_fc(task_size)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.eval_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task > 0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._cur_task > 0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network)
            old_class_mean = self._class_means[: self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args.get("ssca", True):
                old_class_mean += gap
                self._class_means[: self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)

        if self._cur_task > 0 and self.ca_epochs > 0 and self.args.get("ca", True):
            self._stage2_compact_classifier(task_size, self.ca_epochs)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

        if self._cur_task == 0:
            self.tuned_epochs = self.init_epochs
            param_groups = [
                {"params": network.convnet.blocks[-1].parameters(), "lr": 0.01, "weight_decay": self.weight_decay},
                {"params": network.convnet.blocks[:-1].parameters(), "lr": 0.01, "weight_decay": self.weight_decay},
                {"params": network.fc.parameters(), "lr": 0.01, "weight_decay": self.weight_decay},
            ]
        else:
            self.tuned_epochs = self.inc_epochs
            param_groups = [
                {"params": network.convnet.parameters(), "lr": 0.01, "weight_decay": self.weight_decay},
                {"params": network.fc.parameters(), "lr": 0.01, "weight_decay": self.weight_decay},
            ]

        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args["optimizer"] == "adam":
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f"Unsupported optimizer for SSIAT: {self.args['optimizer']}")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.tuned_epochs,
            eta_min=self.min_lr,
        )
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.tuned_epochs))
        loss_cos = AngularPenaltySMLoss(loss_type="cosface", eps=1e-7, s=self.scale, m=self.margin)
        final_info = ""

        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = loss_cos(logits[:, self._known_classes :], targets - self._known_classes)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            avg_loss = losses / len(train_loader)
            lr = optimizer.param_groups[0]["lr"]
            final_info = (
                f"Task {self._cur_task}, Epoch {epoch + 1}/{self.tuned_epochs} => "
                f"Loss {avg_loss:.3f}, Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}"
            )
            prog_bar.set_description(final_info)
            self._record_train_epoch(
                epoch + 1,
                self.tuned_epochs,
                avg_loss,
                train_acc,
                lr=lr,
                test_acc=test_acc,
            )

        logging.info(final_info)

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for param in self._network.fc.parameters():
            param.requires_grad = True

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{"params": param_list, "lr": self.init_lr, "weight_decay": self.weight_decay}]

        optimizer = optim.SGD(network_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.eval()

        for epoch in range(run_epochs):
            losses = 0.0
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device) * (0.9 + decay)
                cls_cov = self._class_covs[c_id].to(self._device)
                distribution = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = distribution.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            inputs = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            shuffle_index = torch.randperm(inputs.size(0))
            inputs = inputs[shuffle_index]
            targets = targets[shuffle_index]

            for iter_id in range(crct_num):
                inp = inputs[iter_id * num_sampled_pcls : (iter_id + 1) * num_sampled_pcls]
                tgt = targets[iter_id * num_sampled_pcls : (iter_id + 1) * num_sampled_pcls]

                if isinstance(self._network, nn.DataParallel):
                    outputs = self._network.module.ca_forward(inp)
                else:
                    outputs = self._network.ca_forward(inp)
                logits = self.scale * outputs["logits"]

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for task_id in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[task_id]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[task_id]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            avg_loss = losses / self._total_classes
            logging.info("CA Task %s => Loss %.3f, Test_accy %.3f", self._cur_task, avg_loss, test_acc)
            self._record_extra_stage_epoch("ca", epoch + 1, run_epochs, loss=avg_loss, test_acc=test_acc)

        if isinstance(self._network, nn.DataParallel):
            self._network = self._network.module

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, "_class_means") and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[: self._known_classes] = self._class_means
            self._class_means = new_class_means

            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[: self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        radius = []
        for class_idx in range(self._known_classes, self._total_classes):
            _, _, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=FEATURE_STATS_BATCH_SIZE,
                shuffle=False,
                num_workers=FEATURE_STATS_WORKERS,
            )
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = np.mean(vectors, axis=0)
            if self._cur_task == 0:
                cov = np.cov(vectors.T) + np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) / 768)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

        if self._cur_task == 0 and radius:
            self.radius = np.sqrt(np.mean(radius))

    def displacement(self, y_old, y_new, embedding_old, sigma):
        delta_y = y_new - y_old
        distance = np.sum(
            (
                np.tile(y_old[None, :, :], [embedding_old.shape[0], 1, 1])
                - np.tile(embedding_old[:, None, :], [1, y_old.shape[0], 1])
            )
            ** 2,
            axis=2,
        )
        weights = np.exp(-distance / (2 * sigma**2)) + 1e-5
        weights = weights / np.tile(np.sum(weights, axis=1)[:, None], [1, weights.shape[1]])
        displacement = np.sum(
            np.tile(weights[:, :, None], [1, 1, delta_y.shape[1]]) * np.tile(delta_y[None, :, :], [weights.shape[0], 1, 1]),
            axis=1,
        )
        return displacement
