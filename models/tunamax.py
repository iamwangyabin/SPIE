import numpy as np
import torch

from models.tuna import Learner as TunaLearner


class Learner(TunaLearner):
    def _train(self, train_loader, test_loader):
        del test_loader
        backbone = self._network.backbone
        backbone_module = self._backbone_module()

        backbone.to(self._device)
        self._network.fc.to(self._device)
        optimizer = self.get_optimizer(backbone)
        scheduler = self.get_scheduler(optimizer)

        self._init_train(train_loader, None, optimizer, scheduler)
        backbone_module.adapter_update()
        self._compute_mean(backbone)
        if self._cur_task > 0:
            self.classifer_align(backbone)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        active_adapter_ids = list(range(self._cur_task + 1))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            logits_per_adapter = []
            with torch.no_grad():
                for adapter_id in active_adapter_ids:
                    features = self._network.backbone(inputs, adapter_id=adapter_id, train=False)["features"]
                    logits = self._network.fc(features)["logits"][:, : self._total_classes] * self.args["scale"]
                    logits_per_adapter.append(logits)

                stacked_logits = torch.stack(logits_per_adapter, dim=0)
                best_scores_per_adapter = stacked_logits.max(dim=2).values.transpose(0, 1)
                best_adapter_per_sample = best_scores_per_adapter.argmax(dim=1)
                selected_logits = stacked_logits.permute(1, 0, 2)[
                    torch.arange(stacked_logits.shape[1], device=stacked_logits.device),
                    best_adapter_per_sample,
                ]

            topk = min(self.topk, selected_logits.shape[1])
            predicts = torch.topk(selected_logits, k=topk, dim=1, largest=True, sorted=True)[1]
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
