import copy
import logging
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import scipy.ndimage
import torch
from PIL import Image
from torch.amp import GradScaler
from torch import Tensor, nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from backbone.linears import SimpleContinualLinear
from backbone.vit_vpt_nsp2pp import VPTNSP2PPBackbone
from models.base import BaseLearner
from utils.mod_adam_vpt import ProjectedModAdam
from utils.supc_loss import SupConLossByGPS
from utils.toolkit import tensor2numpy

num_workers = 8


def _load_pil(image_or_path):
    if isinstance(image_or_path, str):
        with open(image_or_path, "rb") as f:
            return Image.open(f).convert("RGB")
    return Image.fromarray(image_or_path)


class ArrayOrPathDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(_load_pil(self.images[idx]))
        return idx, image, int(self.labels[idx])


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]


class ContrastiveDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = _load_pil(self.images[idx])
        return self.transform(image), int(self.labels[idx])


class RepeatedDataset(Dataset):
    def __init__(self, dataset, repeat: int):
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.dataset = dataset
        self.repeat = int(repeat)

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


class VPTNSP2PPNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone_type = args.get("backbone_type", "vit_base_patch16_224_in21k_vpt_nsp2pp")
        model_name = backbone_type[:-11] if backbone_type.endswith("_vpt_nsp2pp") else backbone_type
        self.backbone = VPTNSP2PPBackbone(
            model_name=model_name,
            pretrained=args.get("pretrained", True),
            prompt_len=args.get("prompt_len", 4),
            prompt_start_block=args.get("prompt_start_block", 0),
            prompt_end_block=args.get("prompt_end_block", 11),
            prompt_init=args.get("prompt_init", "uniform"),
        )
        self.fc = None
        self.feature_dim = self.backbone.out_dim

    def update_fc(self, nb_classes):
        if self.fc is None:
            self.fc = SimpleContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old=True)

    def extract_vector(self, x):
        return self.backbone(x, train=False)["features"]

    def forward(self, x):
        features = self.extract_vector(x)
        outputs = self.fc(features)
        outputs["features"] = features
        return outputs


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = VPTNSP2PPNet(args)
        self.dataset_name = str(args.get("dataset", "")).lower()
        self.augmentation_protocol = str(args.get("augmentation_protocol", "benchmark")).lower()

        self.batch_size = int(args.get("batch_size", 240))
        self.eval_batch_size = int(args.get("eval_batch_size", 100))
        self.num_workers = int(args.get("workers", args.get("num_workers", 16)))
        self.eval_workers = int(args.get("eval_workers", 2))
        self.epochs = int(args.get("epochs", args.get("tuned_epoch", 10)))
        self.expand_times = int(args.get("expand_times", 10))
        self.init_lr = float(args.get("lr", args.get("init_lr", 1e-2)))
        self.weight_decay = float(args.get("weight_decay", 5e-5))
        self.min_lr = float(args.get("min_lr", 1e-5))
        self.temperature = float(args.get("temperature", 30.0))
        self.use_amp = bool(args.get("use_amp", True))

        self.prompt_len = int(args.get("prompt_len", 4))
        self.prompt_start_block = int(args.get("prompt_start_block", 0))
        self.prompt_end_block = int(args.get("prompt_end_block", 11))

        self.use_null_space = bool(args.get("use_null_space", True))
        self.null_thres_mode = args.get("null_thres_mode", "adaptive")
        self.null_thres_value1 = float(args.get("null_thres_value1", 0.0))
        self.null_thres_value2 = float(args.get("null_thres_value2", 0.0))
        self.null_alpha1 = float(args.get("null_alpha1", args.get("null_eta1", 1.0)))
        self.null_alpha2 = float(args.get("null_alpha2", args.get("null_eta2", 1.0)))

        self.impt_enable = bool(args.get("impt_enable", True))
        self.impt_batch_size = int(args.get("impt_batch_size", 100))
        self.impt_topk = float(args.get("impt_topk", 50))
        self.impt_select_level = args.get("impt_select_level", "dime")
        self.impt_more_relax = float(args.get("impt_more_relax", 0.04))
        self.impt_lr_decay = float(args.get("impt_lr_decay", 1.0))
        self.impt_momentum_old = float(args.get("impt_momentum_old", 1.0))
        self.impt_contrast_augment = args.get("impt_contrast_augment", "autoaug")

        self.refine_head = bool(args.get("refine_head", False))
        self.refine_epochs = int(args.get("refine_epochs", 50))
        self.refine_lr = float(args.get("refine_lr", 1e-3))
        self.refine_samples_per_class = int(args.get("refine_samples_per_class", 256))

        self._cached_interm_tensor_dict = {}
        self._update_projection_dict = {}
        self._old_importance_dict = None
        self._mask_dict = {}
        self._class_mean_tensors: Dict[int, Tensor] = {}
        self._class_cov_tensors: Dict[int, Tensor] = {}

        self._configure_trainable_parameters()
        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info("%s VPT-NSP2++ backbone parameters.", f"{total_params:,}")
        logging.info("%s VPT-NSP2++ trainable backbone parameters.", f"{trainable_params:,}")

    def _configure_trainable_parameters(self):
        for _, param in self._network.backbone.named_parameters():
            param.requires_grad = False
        for _, param in self._network.backbone.prompt_items():
            param.requires_grad = True

    def _build_transform(self, training: bool, contrastive: bool = False):
        cfg = getattr(self._network.backbone, "pretrained_cfg", {}) or {}
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        bilinear = T.InterpolationMode.BILINEAR
        official_image_datasets = {"imagenetr", "domainnet", "sdomainet"}
        use_official_aug = self.augmentation_protocol == "official" and self.dataset_name in official_image_datasets
        if contrastive:
            if use_official_aug and self.impt_contrast_augment == "autoaug":
                return T.Compose(
                    [
                        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear),
                        T.Resize((256, 256), interpolation=bilinear, antialias=True),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean, std),
                    ]
                )
            return T.Compose(
                [
                    T.RandomResizedCrop((224, 224), interpolation=bilinear, antialias=True),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                ]
            )
        if training:
            if use_official_aug:
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
                    T.RandomResizedCrop(
                        224,
                        scale=(0.05, 1.0),
                        ratio=(3.0 / 4.0, 4.0 / 3.0),
                        interpolation=bilinear,
                        antialias=True,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                ]
            )
        if use_official_aug:
            return T.Compose(
                [
                    T.Resize((256, 256), interpolation=bilinear, antialias=True),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            )
        return T.Compose(
            [
                T.Resize(256, interpolation=bilinear, antialias=True),
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        )

    def _get_raw_data(self, data_manager, indices, source="train"):
        data, targets, _ = data_manager.get_dataset(indices, source=source, mode="test", ret_data=True)
        return data, targets

    def _make_eval_like_dataset(self, data_manager, indices, source="train"):
        data, targets = self._get_raw_data(data_manager, indices, source=source)
        return ArrayOrPathDataset(data, targets, self._build_transform(training=False))

    def _make_contrastive_dataset(self, data_manager, indices):
        data, targets = self._get_raw_data(data_manager, indices, source="train")
        return ContrastiveDataset(data, targets, TwoCropTransform(self._build_transform(training=True, contrastive=True)))

    def _make_optimizer(self):
        prompt_params = [param for _, param in self._network.backbone.prompt_items()]
        head_params = [p for p in self._network.fc.parameters() if p.requires_grad]
        param_groups = [
            {"params": prompt_params, "lr": self.init_lr, "weight_decay": self.weight_decay},
            {"params": head_params, "lr": self.init_lr, "weight_decay": self.weight_decay},
        ]

        optimizer_name = self.args.get("optimizer", "mod_adam").lower()
        if optimizer_name == "mod_adam":
            optimizer = ProjectedModAdam(param_groups, lr=self.init_lr, weight_decay=self.weight_decay)
            optimizer.update_projection_dict = self._update_projection_dict
            optimizer.mask_dict = self._mask_dict
            optimizer.null_alpha1 = self.null_alpha1
            optimizer.null_alpha2 = self.null_alpha2
            optimizer.impt_more_relax = self.impt_more_relax
            optimizer.impt_lr_decay = self.impt_lr_decay
            return optimizer
        if optimizer_name == "adam":
            return optim.Adam(param_groups)
        if optimizer_name == "adamw":
            return optim.AdamW(param_groups)
        if optimizer_name == "sgd":
            return optim.SGD(param_groups, momentum=0.9)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _make_scheduler(self, optimizer):
        scheduler_name = str(self.args.get("lr_sch", self.args.get("scheduler", "multistep"))).lower()
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.min_lr)
        if scheduler_name in {"steplr", "multistep"}:
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.args.get("decay_milestones", self.args.get("init_milestones", [5, 8])),
                gamma=self.args.get("decay_rate", self.args.get("init_lr_decay", 0.1)),
            )
        if scheduler_name == "constant":
            return None
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _classification_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        current_logits = logits[:, self._known_classes : self._total_classes] / self.temperature
        current_targets = targets - self._known_classes
        return F.cross_entropy(current_logits, current_targets)

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
            zero_idx[max(len(points) - 1, 0) :] = 1
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
        projection_dict = {}
        for pid, named_tensors in interm_tensor_dict.items():
            projection_dict[pid] = {}
            for mname, matrix in named_tensors.items():
                _, singular_values, vt = torch.linalg.svd(matrix, full_matrices=True)
                threshold_value = self.null_thres_value1 if mname == "interm_reader_1" else self.null_thres_value2
                if self.null_thres_mode == "adaptive":
                    zero_idx = self._adaptive_threshold(singular_values, threshold_value)
                elif self.null_thres_mode == "times":
                    zero_idx = singular_values <= singular_values[-1] * int(threshold_value)
                elif self.null_thres_mode == "num":
                    zero_idx = singular_values <= singular_values[-int(threshold_value)]
                elif self.null_thres_mode == "pct":
                    zero_idx = singular_values <= singular_values[-round(threshold_value / 100.0 * singular_values.shape[0])]
                elif self.null_thres_mode == "val":
                    zero_idx = singular_values <= threshold_value
                else:
                    raise ValueError(self.null_thres_mode)
                if torch.count_nonzero(zero_idx) == 0:
                    zero_idx[-1] = True
                basis = vt[zero_idx]
                proj = basis.T @ basis
                projection_dict[pid][mname] = (proj / torch.norm(proj)).detach()
        return projection_dict

    @torch.no_grad()
    def _update_null_space(self):
        if not self.use_null_space or self._cur_task + 1 >= self.data_manager.nb_tasks:
            return

        interm_tensor_dict = {}
        self._network.backbone.eval()
        for _, inputs, _ in self.null_loader:
            inputs = inputs.to(self._device)
            self._network.backbone(inputs, train=False, record_null_stats=interm_tensor_dict)

        if not self._cached_interm_tensor_dict:
            self._cached_interm_tensor_dict = interm_tensor_dict
        else:
            for pid, tensors in interm_tensor_dict.items():
                if pid not in self._cached_interm_tensor_dict:
                    self._cached_interm_tensor_dict[pid] = tensors
                    continue
                for mname, value in tensors.items():
                    if mname not in self._cached_interm_tensor_dict[pid]:
                        self._cached_interm_tensor_dict[pid][mname] = value
                    else:
                        self._cached_interm_tensor_dict[pid][mname] += value
        self._update_projection_dict = self._build_projection_dict(self._cached_interm_tensor_dict)

    def _compute_prompt_importance(self, data_manager, indices):
        dataset = self._make_contrastive_dataset(data_manager, indices)
        loader = DataLoader(dataset, batch_size=self.impt_batch_size, shuffle=False, num_workers=self.eval_workers)
        criterion = SupConLossByGPS(normalize=True).to(self._device)

        self._network.backbone.to(self._device)
        self._network.backbone.train()
        self._network.backbone.zero_grad(set_to_none=True)

        for _, param in self._network.backbone.named_parameters():
            param.requires_grad = False
        for _, param in self._network.backbone.prompt_items():
            param.requires_grad = True

        for images, targets in loader:
            images = torch.cat([images[0], images[1]], dim=0).to(self._device)
            targets = targets.to(self._device)
            features = self._network.backbone(images, train=True)["features"]
            features = features.view(targets.shape[0], 2, -1)
            loss = criterion(features, targets)
            loss.backward()

        importance_dict = OrderedDict()
        for _, param in self._network.backbone.prompt_items():
            importance_dict[id(param)] = param.grad.detach().abs().squeeze(0).cpu()

        self._network.backbone.zero_grad(set_to_none=True)
        self._configure_trainable_parameters()
        return importance_dict

    def _accumulate_old_importance(self, importance_dict):
        if self._old_importance_dict is None or self.impt_momentum_old == 1:
            self._old_importance_dict = importance_dict
            return
        merged = OrderedDict()
        for pid in importance_dict.keys():
            prev = self._old_importance_dict[pid]
            curr = importance_dict[pid]
            merged[pid] = (1 - self.impt_momentum_old) * prev + self.impt_momentum_old * curr
        self._old_importance_dict = merged

    def _select_important_params(self, old_importance_dict, new_importance_dict):
        ordered_pids = [id(param) for _, param in self._network.backbone.prompt_items()]
        stack_old = torch.stack([old_importance_dict[pid] for pid in ordered_pids], dim=0)
        stack_new = torch.stack([new_importance_dict[pid] for pid in ordered_pids], dim=0)
        layers, prompt_len, embed_dim = stack_old.shape

        if self.impt_select_level == "elem":
            num_top = int(stack_old.numel() * self.impt_topk)
            old_idx = stack_old.reshape(-1).topk(num_top).indices
            new_idx = stack_new.reshape(-1).topk(num_top).indices
            mask_old = torch.zeros(stack_old.numel(), dtype=torch.bool)
            mask_new = torch.zeros(stack_new.numel(), dtype=torch.bool)
            mask_old[old_idx] = True
            mask_new[new_idx] = True
            mask_old = mask_old.view(layers, prompt_len, embed_dim)
            mask_new = mask_new.view(layers, prompt_len, embed_dim)
        elif self.impt_select_level == "token":
            num_top = int(self.impt_topk)
            old_idx = stack_old.mean(dim=2).reshape(-1).topk(num_top).indices
            new_idx = stack_new.mean(dim=2).reshape(-1).topk(num_top).indices
            mask_old = torch.zeros(layers * prompt_len, dtype=torch.bool)
            mask_new = torch.zeros(layers * prompt_len, dtype=torch.bool)
            mask_old[old_idx] = True
            mask_new[new_idx] = True
            mask_old = mask_old.view(layers, prompt_len, 1).expand(-1, -1, embed_dim)
            mask_new = mask_new.view(layers, prompt_len, 1).expand(-1, -1, embed_dim)
        elif self.impt_select_level == "dime":
            num_top = int(self.impt_topk)
            old_idx = stack_old.mean(dim=1).reshape(-1).topk(num_top).indices
            new_idx = stack_new.mean(dim=1).reshape(-1).topk(num_top).indices
            mask_old = torch.zeros(layers * embed_dim, dtype=torch.bool)
            mask_new = torch.zeros(layers * embed_dim, dtype=torch.bool)
            mask_old[old_idx] = True
            mask_new[new_idx] = True
            mask_old = mask_old.view(layers, 1, embed_dim).expand(-1, prompt_len, -1)
            mask_new = mask_new.view(layers, 1, embed_dim).expand(-1, prompt_len, -1)
        else:
            raise ValueError(self.impt_select_level)

        self._mask_dict = {}
        mask_intersection = mask_old & mask_new
        mask_nonunionset = ~(mask_old | mask_new)
        mask_only_old = mask_old & ~mask_intersection
        mask_only_new = mask_new & ~mask_intersection
        for pid, mi, mn, mo, mnw in zip(ordered_pids, mask_intersection, mask_nonunionset, mask_only_old, mask_only_new):
            self._mask_dict[pid] = torch.stack([mi, mn, mo, mnw]).to(self._device)

    def _build_safe_distribution(self, mean: Tensor, covariance: Tensor):
        mean = mean.to(self._device, dtype=torch.float32)
        covariance = covariance.to(self._device, dtype=torch.float64)
        covariance = 0.5 * (covariance + covariance.T)
        eye = torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)
        jitter = float(self.args.get("covariance_regularization", 1e-4))
        for power in range(7):
            candidate = covariance + eye * (jitter * (10 ** power))
            _, info = torch.linalg.cholesky_ex(candidate)
            if torch.all(info == 0):
                scale_tril = torch.linalg.cholesky(candidate).to(dtype=torch.float32)
                return MultivariateNormal(mean, scale_tril=scale_tril)
        scale_tril = torch.linalg.cholesky(covariance + eye * max(jitter, 1e-3)).to(dtype=torch.float32)
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def _refresh_class_means_matrix(self):
        if len(self._class_mean_tensors) == 0:
            if hasattr(self, "_class_means"):
                delattr(self, "_class_means")
            return

        matrix = np.zeros((self._total_classes, self.feature_dim), dtype=np.float32)
        for class_idx in range(self._total_classes):
            if class_idx not in self._class_mean_tensors:
                continue
            vector = self._class_mean_tensors[class_idx].detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            matrix[class_idx] = vector
        self._class_means = matrix

    @torch.no_grad()
    def _extract_task_class_stats(self):
        self._network.backbone.eval()
        for class_idx in range(self._known_classes, self._total_classes):
            dataset = self._make_eval_like_dataset(self.data_manager, np.arange(class_idx, class_idx + 1), source="train")
            loader = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.eval_workers)
            features = []
            for _, inputs, _ in loader:
                inputs = inputs.to(self._device)
                features.append(self._network.backbone(inputs, train=False)["features"])
            features = torch.cat(features, dim=0)
            self._class_mean_tensors[class_idx] = features.mean(dim=0).detach()
            self._class_cov_tensors[class_idx] = torch.cov(features.T) + torch.eye(features.shape[-1], device=features.device) * 1e-4
        self._refresh_class_means_matrix()

    def _refine_classifier(self):
        if not self.refine_head or self._cur_task < 0 or len(self._class_mean_tensors) == 0:
            return
        for param in self._network.fc.parameters():
            param.requires_grad = True

        optimizer = optim.SGD(self._network.fc.parameters(), lr=self.refine_lr, weight_decay=1e-4, momentum=0.9)
        criterion = nn.CrossEntropyLoss().to(self._device)
        self._network.fc.to(self._device)
        self._network.fc.train()

        seen_classes = list(range(self._total_classes))
        for _ in range(self.refine_epochs):
            sample_feats = []
            sample_targets = []
            for class_idx in seen_classes:
                dist = self._build_safe_distribution(self._class_mean_tensors[class_idx], self._class_cov_tensors[class_idx])
                feats = dist.sample((self.refine_samples_per_class,))
                tgts = torch.full((self.refine_samples_per_class,), class_idx, dtype=torch.long, device=self._device)
                sample_feats.append(feats)
                sample_targets.append(tgts)

            features = torch.cat(sample_feats, dim=0)
            targets = torch.cat(sample_targets, dim=0)
            perm = torch.randperm(features.shape[0], device=self._device)
            features = features[perm]
            targets = targets[perm]

            logits = self._network.fc(features)["logits"][:, : self._total_classes] / self.temperature
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._reset_task_logging()
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        self.data_manager = data_manager
        self._network.update_fc(task_size)

        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        current_indices = np.arange(self._known_classes, self._total_classes)
        seen_indices = np.arange(0, self._total_classes)

        base_train_dataset = data_manager.get_dataset(current_indices, source="train", mode="train")
        self.train_dataset = RepeatedDataset(base_train_dataset, self.expand_times)
        self.test_dataset = data_manager.get_dataset(seen_indices, source="test", mode="test")
        self.null_dataset = self._make_eval_like_dataset(data_manager, current_indices, source="train")

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.eval_workers)
        self.null_loader = DataLoader(self.null_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.eval_workers)

        if self._cur_task > 0 and self.impt_enable and self._old_importance_dict is not None:
            new_importance_dict = self._compute_prompt_importance(data_manager, current_indices)
            self._select_important_params(self._old_importance_dict, new_importance_dict)
        else:
            self._mask_dict = {}

        self._train(self.train_loader)
        self._extract_task_class_stats()
        self._refine_classifier()
        self._update_null_space()

        if self.impt_enable and self._cur_task < data_manager.nb_tasks - 1:
            trained_importance = self._compute_prompt_importance(data_manager, current_indices)
            self._accumulate_old_importance(trained_importance)

    def _train(self, train_loader):
        self._network.to(self._device)
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer)
        scaler = GradScaler(device="cuda", enabled=self.use_amp and self._device.type == "cuda")

        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
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
                    outputs = self._network(inputs)
                    logits = outputs["logits"]
                    loss = self._classification_loss(logits, targets)

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

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp and self._device.type == "cuda",
                ):
                    outputs = self._network(inputs)["logits"][:, : self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _extract_vectors(self, loader):
        self._network.eval()
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
