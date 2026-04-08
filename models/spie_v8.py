import copy
import logging

import torch

from models.spie_v7 import Learner as SPiEV7Learner


class Learner(SPiEV7Learner):
    """SPiE v8 learner with supervised shared-LoRA delta EMA updates."""

    _spie_version_name = "SPiE v8"

    def __init__(self, args):
        super().__init__(args)
        self.task0_copy_shared_to_expert = False
        self._freeze_unused_shared_predictor()
        self.shared_update_epochs = int(args.get("shared_update_epochs", 3))
        self.shared_update_lr = float(
            args.get("shared_update_lr", self.init_lr * args.get("shared_update_lr_scale", 0.02))
        )
        self.shared_ema_alpha = float(args.get("shared_ema_alpha", 0.05))

        if not 0.0 <= self.shared_ema_alpha <= 1.0:
            raise ValueError(f"shared_ema_alpha must be in [0, 1], got {self.shared_ema_alpha}")

        logging.info(
            "SPiE v8 shared update: epochs=%s, lr=%s, ema_alpha=%s.",
            self.shared_update_epochs,
            self.shared_update_lr,
            self.shared_ema_alpha,
        )

    def _freeze_unused_shared_predictor(self):
        backbone = self._backbone_module()
        if hasattr(backbone, "cassle_predictor"):
            backbone.cassle_predictor.requires_grad_(False)

    def _swap_in_temporary_fc(self):
        main_fc = self._network.fc
        temporary_fc = copy.deepcopy(main_fc).to(self._device)
        for param in temporary_fc.parameters():
            param.requires_grad = True
        self._network.fc = temporary_fc
        return main_fc

    def _restore_main_fc(self, main_fc):
        self._network.fc = main_fc
        self._network.fc.to(self._device)

    def _shared_delta_optimizer(self):
        backbone = self._backbone_module()
        network_params = [
            {
                "params": [p for p in backbone.cur_shared_adapter.parameters() if p.requires_grad],
                "lr": self.shared_update_lr,
                "weight_decay": self.share_lora_weight_decay,
            },
            {
                "params": self._network.fc.parameters(),
                "lr": self.init_lr,
                "weight_decay": self.weight_decay,
            },
        ]
        return self._make_optimizer(network_params)

    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        if self._cur_task == 0:
            self._train_task0_shared_lora(train_loader)
            self._train_task0_expert(train_loader)
        else:
            self._train_incremental_expert(train_loader)
            self._train_shared_delta(train_loader)

        self._freeze_shared_domain_adapter()
        self._set_current_expert_requires_grad(True)
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

    def _train_task0_shared_lora(self, train_loader):
        if self.task0_shared_epochs <= 0:
            return

        main_fc = self._swap_in_temporary_fc()
        try:
            self._set_shared_lora_requires_grad(True)
            self._set_current_expert_requires_grad(False)
            optimizer = self._task0_shared_optimizer()
            scheduler = self._get_scheduler_for_epochs(optimizer, self.task0_shared_epochs)
            self._run_task0_phase(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.task0_shared_epochs,
                adapter_id=-1,
                stage="task0_shared_lora",
            )
        finally:
            self._restore_main_fc(main_fc)

    def _train_shared_delta(self, train_loader):
        if self.shared_update_epochs <= 0 or self.shared_ema_alpha <= 0.0:
            return

        backbone = self._backbone_module()
        main_fc = self._swap_in_temporary_fc()
        original_shared_adapter = None
        try:
            original_shared_adapter = backbone.cur_shared_adapter
            shared_work = copy.deepcopy(original_shared_adapter).to(self._device)

            original_shared_adapter.requires_grad_(False)
            self._set_current_expert_requires_grad(False)
            shared_work.requires_grad_(True)
            backbone.cur_shared_adapter = shared_work

            optimizer = self._shared_delta_optimizer()
            scheduler = self._get_scheduler_for_epochs(optimizer, self.shared_update_epochs)
            self._run_task0_phase(
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self.shared_update_epochs,
                adapter_id=-1,
                stage="shared_delta",
            )
            self._ema_update_shared_adapter(original_shared_adapter, shared_work)
        finally:
            if original_shared_adapter is not None:
                backbone.cur_shared_adapter = original_shared_adapter
            self._restore_main_fc(main_fc)
            self._freeze_shared_domain_adapter()

    @torch.no_grad()
    def _ema_update_shared_adapter(self, target_shared_adapter, source_shared_adapter):
        for target_param, source_param in zip(target_shared_adapter.parameters(), source_shared_adapter.parameters()):
            source_data = source_param.data.to(device=target_param.device, dtype=target_param.dtype)
            target_param.data.add_(source_data - target_param.data, alpha=self.shared_ema_alpha)
