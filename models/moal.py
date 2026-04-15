import copy
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from backbone.linears import CosineLinear
from models.base import BaseLearner
from utils.moal_net import MoALNet, SimpleVitNetAL
from utils.toolkit import target2onehot, tensor2numpy

num_workers = 8


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        if x.dtype != self.fc.weight.dtype:
            x = x.to(dtype=self.fc.weight.dtype)
        return self.fc(x)


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return idx, self.features[idx], self.labels[idx]


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if "_moal" not in args["backbone_type"]:
            raise NotImplementedError("MoAL requires a backbone type ending with '_moal'.")

        self._network = SimpleVitNetAL(args, True)
        self.batch_size = int(args["batch_size"])
        self.init_lr = float(args["init_lr"])
        self.progressive_lr = float(args["progressive_lr"])
        self.model_hidden = int(args["Hidden"])
        self.weight_decay = float(args["weight_decay"] if args["weight_decay"] is not None else 0.0005)
        self.min_lr = float(args["min_lr"] if args["min_lr"] is not None else 1e-8)
        self.alpha = float(args.get("alpha", 0.999))
        self.args = args
        self.R = None
        self._means = []
        self._cov_matrix = []
        self._std_deviations_matrix = []

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        self.old_network_module_ptr = self._old_network.module if hasattr(self._old_network, "module") else self._old_network

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        if self._cur_task == 0:
            self._network.fc = CosineLinear(self._network.feature_dim, self._total_classes).to(self._device)

        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset,
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

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0:
            optimizer = self._build_optimizer(self._network.parameters(), self.init_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args["tuned_epoch"],
                eta_min=self.min_lr,
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.construct_dual_branch_network()
            self._network.update_fc(self._total_classes, self.model_hidden)
        else:
            self._network.update_fc(self._total_classes, self.model_hidden, cosine_fc=True)
            self._network.update_fc(self._total_classes, self.model_hidden)
            for param in self._network.ac_model.parameters():
                param.requires_grad = False
            optimizer = self._build_optimizer(self._network.parameters(), self.progressive_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args["progreesive_epoch"],
                eta_min=self.min_lr,
            )
            self._progressive_train(train_loader, test_loader, optimizer, scheduler)

        if self._cur_task == 0:
            self._compute_means()
            self._network.to(self._device)
            self.cls_align(train_loader, self._network)
        else:
            self._compute_means()
            self.cali_prototye_model(train_loader)
            self._compute_relations()
            self._build_feature_set()
            self._network.to(self._device)
            self.IL_align(train_loader, self._network)
            self.cali_weight(self._feature_trainset, self._network)

    def _build_optimizer(self, params, lr):
        trainable_params = [param for param in params if param.requires_grad]
        if self.args["optimizer"] == "sgd":
            return optim.SGD(trainable_params, momentum=0.9, lr=lr, weight_decay=self.weight_decay)
        if self.args["optimizer"] == "adam":
            return optim.AdamW(trainable_params, lr=lr, weight_decay=self.weight_decay)
        raise ValueError("Unsupported optimizer {}".format(self.args["optimizer"]))

    def construct_dual_branch_network(self):
        network = MoALNet(self.args, True)
        network.construct_dual_branch_network(self._network)
        self._network = network.to(self._device)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            avg_loss = losses / len(train_loader)
            lr = optimizer.param_groups[0]["lr"]

            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.args["tuned_epoch"],
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}"
            ).format(self._cur_task, epoch + 1, self.args["tuned_epoch"], avg_loss, train_acc, test_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _progressive_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["progreesive_epoch"]))
        ema_model = self._network.copy().freeze()

        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["train_logits"]
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                preds = torch.max(logits, dim=1)[1]
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            for param, ema_param in zip(self._network.backbones[0].parameters(), ema_model.backbones[0].parameters()):
                ema_param.data = self.alpha * ema_param.data + (1 - self.alpha) * param.data

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_ac_train_accuracy(self._network, test_loader)
            avg_loss = losses / len(train_loader)
            lr = optimizer.param_groups[0]["lr"]

            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.args["progreesive_epoch"],
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
                stage="progressive",
            )
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}"
            ).format(self._cur_task, epoch + 1, self.args["progreesive_epoch"], avg_loss, train_acc, test_acc)
            prog_bar.set_description(info)

        for ema_param, param in zip(ema_model.backbones[0].parameters(), self._network.backbones[0].parameters()):
            param.data = ema_param.data

        logging.info(info)

    def _compute_ac_train_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["train_logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def cls_align(self, trainloader, model):
        model = model.module if hasattr(model, "module") else model
        model = model.eval()
        embedding_list, label_list = [], []

        auto_cor = torch.zeros(model.ac_model.fc[-1].weight.size(1), model.ac_model.fc[-1].weight.size(1)).to(self._device)
        crs_cor = torch.zeros(model.ac_model.fc[-1].weight.size(1), self._total_classes).to(self._device)

        with torch.no_grad():
            for _, data, label in tqdm(trainloader, desc="Alignment", total=len(trainloader), unit="batch"):
                images = data.to(self._device)
                target = label.to(self._device)
                label_list.append(target.cpu())

                feature = model(images)["features"]
                new_activation = model.ac_model.fc[:2](feature)
                embedding_list.append(new_activation.cpu())

                label_onehot = F.one_hot(target, self._total_classes).float()
                auto_cor += torch.t(new_activation) @ new_activation
                crs_cor += torch.t(new_activation) @ label_onehot

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        Y = target2onehot(label_list, self._total_classes)

        ridge = self.optimise_ridge_parameter(embedding_list, Y)
        logging.info("gamma %s", ridge)

        gram = auto_cor.float() + ridge * torch.eye(
            model.ac_model.fc[-1].weight.size(1),
            device=self._device,
            dtype=auto_cor.dtype,
        )
        R = torch.linalg.inv(gram).float()
        delta = R @ crs_cor
        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(0.9 * delta.float()))
        self.R = R

    def optimise_ridge_parameter(self, features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(features.shape[0] * 0.8)
        losses = []
        Q_val = features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = features[0:num_val_samples, :].T @ features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val).T
            Y_train_pred = features[num_val_samples:, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples:, :]))
        return ridges[np.argmin(np.array(losses))]

    def IL_align(self, trainloader, model):
        model = model.module if hasattr(model, "module") else model
        model = model.eval()

        W = model.ac_model.fc[-1].weight.t().float()
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            for _, data, label in tqdm(trainloader, desc="Alignment", total=len(trainloader), unit="batch"):
                images = data.to(self._device)
                target = label.to(self._device)

                feature = model(images)["features"]
                new_activation = model.ac_model.fc[:2](feature)
                label_onehot = F.one_hot(target, self._total_classes).float()

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) + new_activation @ R @ new_activation.t()
                ) @ new_activation @ R

                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)

        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        self.R = R

    def _compute_means(self):
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, _, idx_dataset = self.data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
                vectors, _ = self.extract_prototype(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                self._means.append(class_mean)

                cov = np.cov(vectors, rowvar=False)
                self._cov_matrix.append(cov)
                self._std_deviations_matrix.append(np.sqrt(np.diagonal(cov)))

    def _compute_relations(self):
        old_means = np.array(self._means[: self._known_classes])
        new_means = np.array(self._means[self._known_classes :])
        self._relations = np.argmax(
            (old_means / np.linalg.norm(old_means, axis=1)[:, None])
            @ (new_means / np.linalg.norm(new_means, axis=1)[:, None]).T,
            axis=1,
        ) + self._known_classes

    def extract_prototype(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(self._network(_inputs.to(self._device))["features"])
            vectors.append(_vectors)
            targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    def _build_feature_set(self):
        vectors_train, labels_train = [], []
        for class_idx in range(self._known_classes, self._total_classes):
            data, _, idx_dataset = self.data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self.extract_prototype(idx_loader)
            vectors_train.append(vectors)
            labels_train.append([class_idx] * len(vectors))

        for class_idx in range(0, self._known_classes):
            new_idx = self._relations[class_idx]
            vectors_train.append(vectors_train[new_idx - self._known_classes] - self._means[new_idx] + self._means[class_idx])
            labels_train.append([class_idx] * len(vectors_train[-1]))

        vectors_train = np.concatenate(vectors_train)
        labels_train = np.concatenate(labels_train)
        self._feature_trainset = DataLoader(
            FeatureDataset(vectors_train, labels_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def cali_weight(self, cali_pseudo_feature, model):
        model = model.module if hasattr(model, "module") else model
        model = model.eval()

        W = model.ac_model.fc[-1].weight.t().float()
        R = copy.deepcopy(self.R.float())

        with torch.no_grad():
            for _, data, label in tqdm(cali_pseudo_feature, desc="Alignment", total=len(cali_pseudo_feature), unit="batch"):
                features = data.to(self._device)
                target = label.to(self._device)

                new_activation = model.ac_model.fc[:2](features.float())
                label_onehot = F.one_hot(target, self._total_classes).float()
                output = model.ac_model.fc[-1](new_activation)
                pred = output.topk(1, 1, True, True)[1].t()
                correct = pred.eq(target.view(1, -1).expand_as(pred))
                false_indices = (correct == False).view(-1).nonzero(as_tuple=False)
                if false_indices.numel() == 0:
                    continue

                new_activation = new_activation[false_indices[:, 0]]
                label_onehot = label_onehot[false_indices[:, 0]]

                R = R - R @ new_activation.t() @ torch.pinverse(
                    torch.eye(new_activation.size(0)).to(self._device) + new_activation @ R @ new_activation.t()
                ) @ new_activation @ R

                W = W + R @ new_activation.t() @ (label_onehot - new_activation @ W)

        model.ac_model.fc[-1].weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        self.R = R

    def cali_prototye_model(self, train_loader):
        with torch.no_grad():
            old_vectors, vectors = [], []
            for _, data, _ in tqdm(train_loader, desc="cali_prototye_model", total=len(train_loader), unit="batch"):
                images = data.to(self._device)
                old_feature = tensor2numpy(self.old_network_module_ptr(images)["features"])
                feature = tensor2numpy(self._network(images)["features"])
                old_vectors.append(old_feature)
                vectors.append(feature)

        E_old = np.concatenate(old_vectors)
        E_new = np.concatenate(vectors)
        X_tensor = torch.from_numpy(E_old).to(torch.float32)
        y_tensor = torch.from_numpy(E_new).to(torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)

        total_size = len(dataset)
        train_size = int(0.9 * total_size)
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        calimodel = SimpleNN(E_old[0].shape[0], E_new[0].shape[0]).to(self._device)
        optimizer = optim.SGD(calimodel.parameters(), momentum=0.9, lr=0.01, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        best_loss = float("inf")
        best_model_wts = None

        for _ in tqdm(range(1000)):
            calimodel.train()
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = calimodel(inputs)
                loss = nn.MSELoss()(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            calimodel.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    logits = calimodel(inputs)
                    test_loss += nn.MSELoss()(logits, targets).item() * inputs.size(0)

            test_loss /= len(test_dataset)
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_wts = copy.deepcopy(calimodel.state_dict())

        logging.info("best_loss: %s", best_loss)
        calimodel.load_state_dict(best_model_wts)
        calimodel.eval()

        X_test = torch.from_numpy(np.array(self._means)[: self._known_classes]).to(torch.float64)
        Y_test = calimodel(X_test.to(self._device)).to("cpu").detach().numpy().tolist()
        self._means[: self._known_classes] = Y_test
