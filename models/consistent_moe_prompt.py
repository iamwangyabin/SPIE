import copy
import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.ndimage
import torch
from PIL import Image
from torch import Tensor, nn, optim
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from backbone.linears import SimpleContinualLinear
from models.base import BaseLearner
from utils.inc_net import TUNANet
from utils.toolkit import tensor2numpy

num_workers = 8


class PathTransformDataset(Dataset):
    def __init__(self, paths, labels, transform, expand_times: int = 1):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.expand_times = max(int(expand_times), 1)

    def __len__(self):
        return len(self.paths) * self.expand_times

    def __getitem__(self, idx):
        idx = idx % len(self.paths)
        with open(self.paths[idx], "rb") as f:
            image = Image.open(f).convert("RGB")
        return idx, self.transform(image), int(self.labels[idx])


class ProjectedAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.update_projection_dict: Dict[int, Dict[str, Tensor]] = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if weight_decay != 0:
                    grad = grad.add(p.detach(), alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = (-lr / bias_correction1) * exp_avg / denom

                projection = self.update_projection_dict.get(id(p))
                if projection:
                    if "interm_reader_1" in projection:
                        update = projection["interm_reader_1"] @ update
                    elif "interm_reader_2" in projection:
                        m, l, d = update.shape
                        update = (projection["interm_reader_2"] @ update.view(m, l * d)).view(m, l, d)

                with torch.no_grad():
                    p.add_(update)
        return loss


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = TUNANet(args, True)
        self.args = args
        self.batch_size = int(args["batch_size"])
        self.epochs = int(args["tuned_epoch"])
        self.init_lr = float(args["init_lr"])
        self.weight_decay = float(args.get("weight_decay", 5e-5))
        self.min_lr = float(args.get("min_lr", 1e-5))
        self.temperature = float(args.get("temperature", 30.0))
        self.training_string = tuple(args.get("training_string", ["w_gate", "w_noise", "experts"]))
        self.null_patterns = tuple(args.get("null_patterns", ["experts", "w_gate"]))
        self.use_null_space = bool(args.get("use_null_space", True))
        self.null_thres_mode = args.get("null_thres_mode", "adaptive")
        self.null_thres_value1 = float(args.get("null_thres_value1", 0.0))
        self.null_thres_value2 = float(args.get("null_thres_value2", 0.0))
        self.null_eta1 = float(args.get("null_eta1", 1.0))
        self.null_eta2 = float(args.get("null_eta2", 0.99))
        self.moe_impt_coef = float(args.get("moe_impt_coef", 0.05))
        self.eval_batch_size = int(args.get("eval_batch_size", 100))
        self.eval_workers = int(args.get("eval_workers", 2))
        self.expand_times = int(args.get("expand_times", 10))
        self.use_amp = bool(args.get("use_amp", True))
        self._cached_interm_tensor_dict = {}
        self._update_projection_dict = {}

        self._configure_trainable_parameters()
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s CPG backbone parameters.", f"{total_params:,}")
        logging.info("%s CPG trainable backbone parameters.", f"{trainable_params:,}")

    def _configure_trainable_parameters(self):
        for name, param in self._network.backbone.named_parameters():
            param.requires_grad = any(token in name for token in self.training_string)

    def _build_transforms(self, training: bool):
        cfg = getattr(self._network.backbone, "pretrained_cfg", {}) or {}
        mean = cfg.get("mean", (0.5, 0.5, 0.5))
        std = cfg.get("std", (0.5, 0.5, 0.5))
        bilinear = T.InterpolationMode.BILINEAR
        if training:
            return T.Compose(
                [
                    T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear),
                    T.RandomResizedCrop((224, 224), interpolation=bilinear, antialias=True),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            )
        return T.Compose(
            [
                T.Resize((256, 256), interpolation=bilinear, antialias=True),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    def _subset_dataset(self, data_manager, indices: np.ndarray, source: str, training: bool):
        data, targets, _ = data_manager.get_dataset(indices, source=source, mode="test", ret_data=True)
        expand_times = self.expand_times if training else 1
        return PathTransformDataset(data, targets, self._build_transforms(training), expand_times=expand_times)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        if self._network.fc is None:
            self._network.fc = SimpleContinualLinear(self._network.feature_dim, task_size)
        else:
            self._network.fc.update(task_size, freeze_old=True)
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        current_indices = np.arange(self._known_classes, self._total_classes)
        seen_indices = np.arange(0, self._total_classes)
        self.data_manager = data_manager
        self.train_dataset = self._subset_dataset(data_manager, current_indices, "train", True)
        self.test_dataset = self._subset_dataset(data_manager, seen_indices, "test", False)
        self.null_dataset = self._subset_dataset(data_manager, current_indices, "train", False)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.eval_workers
        )
        self.null_loader = DataLoader(
            self.null_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.eval_workers
        )
        self._train(self.train_loader, self.test_loader)

    def _make_optimizer(self):
        backbone_params = [p for p in self._network.backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": self.init_lr, "weight_decay": self.weight_decay},
            {"params": self._network.fc.parameters(), "lr": self.init_lr, "weight_decay": self.weight_decay},
        ]
        optimizer_name = self.args.get("optimizer", "mod_adam")
        if optimizer_name == "mod_adam":
            optimizer = ProjectedAdam(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)
            optimizer.update_projection_dict = self._update_projection_dict
            return optimizer
        if optimizer_name == "adam":
            return optim.Adam(param_groups)
        if optimizer_name == "adamw":
            return optim.AdamW(param_groups)
        if optimizer_name == "sgd":
            return optim.SGD(param_groups, momentum=0.9)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _make_scheduler(self, optimizer):
        scheduler_name = self.args.get("scheduler", "steplr")
        if scheduler_name == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.args.get("init_milestones", [5, 8]),
                gamma=self.args.get("init_lr_decay", 0.1),
            )
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)
        if scheduler_name == "constant":
            return None
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _classification_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        current_logits = logits[:, self._known_classes : self._total_classes] / self.temperature
        current_targets = targets - self._known_classes
        return F.cross_entropy(current_logits, current_targets)

    def _get_param_id_dict(self):
        param_id_dict = {}
        for name, param in self._network.backbone.named_parameters():
            if param.requires_grad and any(pattern in name for pattern in self.null_patterns):
                param_id_dict[id(param)] = {"name": name, "shape": list(param.shape)}
        return param_id_dict

    def _get_interm_tensor_dict(self):
        param_id_dict = self._get_param_id_dict()
        interm_tensor_dict = {}

        def _forward_hook(module: nn.Module, args: Tuple[Tensor], output: Tensor):
            if not hasattr(module, "module_name"):
                return
            pid = module.dst_param_id
            mname = module.module_name
            if pid not in param_id_dict or mname not in {"interm_reader_1", "interm_reader_2"}:
                return
            interm = args[0].detach().clone()
            interm = torch.matmul(interm.T, interm) / interm.shape[0]
            if pid not in interm_tensor_dict:
                interm_tensor_dict[pid] = {}
            if mname not in interm_tensor_dict[pid]:
                interm_tensor_dict[pid][mname] = torch.zeros_like(interm)
            interm_tensor_dict[pid][mname] += interm

        handle_list = []
        for _, module in self._network.backbone.named_modules():
            if hasattr(module, "module_name") and "interm_reader" in module.module_name:
                handle_list.append(module.register_forward_hook(_forward_hook))

        self._network.backbone.eval()
        with torch.no_grad():
            for _, inputs, _ in self.null_loader:
                inputs = inputs.to(self._device)
                self._network.backbone(inputs, train=False)

        for handle in handle_list:
            handle.remove()
        return interm_tensor_dict

    def _adaptive_threshold(self, singular_values: Tensor, offset: float = 0.0):
        points = singular_values.detach().cpu().numpy()
        if len(points) >= 128:
            fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
            diff_o1 = fil_points[:-1] - fil_points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            drop_num = int(len(points) * 0.03 / 2)
            valid_o2 = diff_o2[drop_num:-drop_num]
            threshold_value = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
        elif len(points) > 3:
            diff_o1 = points[:-1] - points[1:]
            diff_o2 = diff_o1[:-1] - diff_o1[1:]
            threshold_value = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
        else:
            zero_idx = np.zeros(len(points), dtype=np.int64)
            zero_idx[points > 1e-3] = 1
            return torch.as_tensor(zero_idx, dtype=torch.bool, device=singular_values.device)

        i_threshold = np.arange(len(points))[points >= threshold_value].max()
        if 0 <= offset < 1:
            i_threshold = min(i_threshold + int(offset * (len(points) - i_threshold)), len(points) - 1)
        else:
            i_threshold = max(min(i_threshold + int(offset), len(points) - 1), 0)
        zero_idx = np.zeros(len(points), dtype=np.int64)
        zero_idx[i_threshold:] = 1
        return torch.as_tensor(zero_idx, dtype=torch.bool, device=singular_values.device)

    def _build_projection_dict(self, interm_tensor_dict):
        update_proj_dict = {}
        for pid, named_tensors in interm_tensor_dict.items():
            update_proj_dict[pid] = {}
            for mname, matrix in named_tensors.items():
                _, singular_values, vt = torch.linalg.svd(matrix, full_matrices=True)
                threshold_value = self.null_thres_value1 if mname == "interm_reader_1" else self.null_thres_value2
                if self.null_thres_mode == "adaptive":
                    zero_idx = self._adaptive_threshold(singular_values, threshold_value)
                else:
                    zero_idx = singular_values <= singular_values[-1] * int(threshold_value)
                basis = vt[zero_idx]
                proj = basis.T @ basis
                proj = proj / torch.norm(proj)
                eta = self.null_eta1 if mname == "interm_reader_1" else self.null_eta2
                if eta != 1:
                    proj = eta * proj + (1 - eta) * torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)
                update_proj_dict[pid][mname] = proj.detach()
        return update_proj_dict

    def _update_null_space(self):
        if not self.use_null_space or self._cur_task + 1 >= self.data_manager.nb_tasks:
            return
        new_interm_tensor_dict = self._get_interm_tensor_dict()
        if not self._cached_interm_tensor_dict:
            self._cached_interm_tensor_dict = new_interm_tensor_dict
        else:
            for pid, tensors in new_interm_tensor_dict.items():
                if pid not in self._cached_interm_tensor_dict:
                    self._cached_interm_tensor_dict[pid] = tensors
                    continue
                for mname, value in tensors.items():
                    if mname not in self._cached_interm_tensor_dict[pid]:
                        self._cached_interm_tensor_dict[pid][mname] = value
                    else:
                        self._cached_interm_tensor_dict[pid][mname] += value
        self._update_projection_dict = self._build_projection_dict(self._cached_interm_tensor_dict)

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)
        self._network.fc.to(self._device)
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)
        scaler = GradScaler(enabled=self.use_amp and self._device.type == "cuda")

        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp and self._device.type == "cuda",
                ):
                    outputs = self._network.backbone(inputs, train=True)
                    logits = self._network.fc(outputs["features"])["logits"]
                    loss = self._classification_loss(logits, targets)

                    moe_loss = outputs.get("moe_loss")
                    if moe_loss is not None:
                        loss = loss + self.moe_impt_coef * moe_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses += loss.item()

                preds = torch.max(logits[:, : self._total_classes], dim=1)[1]
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_loss = losses / len(train_loader)
            lr = optimizer.param_groups[0]["lr"]
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, self.epochs, avg_loss, train_acc
            )
            self._record_train_epoch(
                epoch=epoch + 1,
                total_epochs=self.epochs,
                loss=float(avg_loss),
                acc=float(train_acc),
                lr=float(lr),
            )
            prog_bar.set_description(info)

        logging.info(info)
        self._update_null_space()

    def _eval_cnn(self, loader):
        self._network.backbone.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp and self._device.type == "cuda",
                ):
                    features = self._network.backbone(inputs, train=False)["features"]
                    outputs = self._network.fc(features)["logits"][:, : self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _extract_vectors(self, loader):
        self._network.backbone.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, inputs, _targets in loader:
                inputs = inputs.to(self._device)
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp and self._device.type == "cuda",
                ):
                    features = self._network.backbone(inputs, train=False)["features"]
                vectors.append(tensor2numpy(features))
                targets.append(_targets.numpy())
        return np.concatenate(vectors), np.concatenate(targets)
