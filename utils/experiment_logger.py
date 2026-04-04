from typing import Any, Dict, Iterable, Optional


def _sanitize_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "default"
    return text.replace(" ", "-").replace("/", "-")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if isinstance(value, set):
        return sorted(_to_builtin(item) for item in value)

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    try:
        import torch

        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
    except ImportError:
        pass

    return value


class ExperimentLogger:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.enabled = bool(args.get("wandb", False))
        self.run = None

        if not self.enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging was enabled, but the 'wandb' package is not installed."
            ) from exc

        self._wandb = wandb
        config = _to_builtin(args)
        group = args.get("wandb_group") or self._default_group(args)
        name = args.get("wandb_name") or self._default_run_name(args)
        tags = args.get("wandb_tags", [])

        init_kwargs = {
            "project": args.get("wandb_project", "SPIE"),
            "config": config,
            "group": group,
            "name": name,
            "tags": tags,
            "mode": args.get("wandb_mode", "online"),
        }
        if args.get("wandb_entity"):
            init_kwargs["entity"] = args["wandb_entity"]

        self.run = wandb.init(**init_kwargs)
        wandb.define_metric("train/global_epoch")
        wandb.define_metric("train/*", step_metric="train/global_epoch")
        wandb.define_metric("extra/global_epoch")
        wandb.define_metric("extra/*", step_metric="extra/global_epoch")
        wandb.define_metric("eval/task_id")
        wandb.define_metric("eval/*", step_metric="eval/task_id")

    def _default_group(self, args: Dict[str, Any]) -> str:
        return "-".join(
            [
                _sanitize_text(args.get("model_name", "model")),
                _sanitize_text(args.get("dataset", "dataset")),
                _sanitize_text(args.get("backbone_type", "backbone")),
                f"init{args.get('init_cls', 'na')}",
                f"inc{args.get('increment', 'na')}",
            ]
        )

    def _default_run_name(self, args: Dict[str, Any]) -> str:
        prefix = _sanitize_text(args.get("prefix", "run"))
        return "-".join(
            [
                prefix,
                f"seed{args.get('seed', 'na')}",
            ]
        )

    def log(self, metrics: Optional[Dict[str, Any]]) -> None:
        if not self.enabled or not metrics:
            return
        self._wandb.log(_to_builtin(metrics))

    def log_train_history(self, task_id: int, train_history: Iterable[Dict[str, Any]]) -> None:
        for item in train_history:
            global_epoch = task_id * item["total_epochs"] + item["epoch"]
            payload = {
                "train/global_epoch": global_epoch,
                "train/task_id": task_id,
                "train/epoch": item["epoch"],
                "train/epoch_in_task": item["epoch"],
                "train/total_epochs": item["total_epochs"],
                "train/known_classes": item.get("known_classes"),
                "train/total_classes": item.get("total_classes"),
            }
            for key, value in item.items():
                if key in {"epoch", "total_epochs", "known_classes", "total_classes"}:
                    continue
                payload[f"train/{key}"] = value
            self.log(payload)

    def log_extra_history(self, task_id: int, extra_history: Iterable[Dict[str, Any]]) -> None:
        for item in extra_history:
            stage = item["stage"]
            total_epochs = item["total_epochs"]
            global_epoch = task_id * total_epochs + item["epoch"]
            payload = {
                "extra/global_epoch": global_epoch,
                "extra/task_id": task_id,
                "extra/stage_epoch": item["epoch"],
                "extra/total_epochs": total_epochs,
                "extra/known_classes": item.get("known_classes"),
                "extra/total_classes": item.get("total_classes"),
            }
            for key, value in item.items():
                if key in {"stage", "epoch", "total_epochs", "known_classes", "total_classes"}:
                    continue
                payload[f"extra/{stage}/{key}"] = value
            self.log(payload)

    def log_eval(
        self,
        task_id: int,
        cnn_accy: Dict[str, Any],
        nme_accy: Optional[Dict[str, Any]],
        avg_cnn: Optional[float] = None,
        avg_nme: Optional[float] = None,
        all_params: Optional[int] = None,
        trainable_params: Optional[int] = None,
    ) -> None:
        payload = {
            "eval/task_id": task_id,
        }
        if all_params is not None:
            payload["eval/model/all_params"] = all_params
        if trainable_params is not None:
            payload["eval/model/trainable_params"] = trainable_params

        payload.update(self._flatten_eval_metrics("cnn", cnn_accy))
        if avg_cnn is not None:
            payload["eval/cnn/avg_top1"] = avg_cnn

        if nme_accy is not None:
            payload.update(self._flatten_eval_metrics("nme", nme_accy))
            if avg_nme is not None:
                payload["eval/nme/avg_top1"] = avg_nme

        self.log(payload)

    def _flatten_eval_metrics(self, prefix: str, accy: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            f"eval/{prefix}/top1": accy["top1"],
            f"eval/{prefix}/top5": accy["top5"],
        }
        grouped = accy.get("grouped", {})
        for key in ("total", "old", "new"):
            if key in grouped:
                payload[f"eval/{prefix}/{key}"] = grouped[key]
        for key, value in grouped.items():
            if key in {"total", "old", "new"}:
                continue
            payload[f"eval/{prefix}/group/{key}"] = value
        return payload

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        if not metrics:
            return
        self.log(metrics)
        if self.enabled and self.run is not None:
            for key, value in _to_builtin(metrics).items():
                self.run.summary[key] = value

    def log_accuracy_matrix(self, prefix: str, matrix, column_labels) -> None:
        if matrix is None or len(matrix) == 0:
            return

        builtin_matrix = _to_builtin(matrix)
        builtin_labels = _to_builtin(column_labels)
        summary_metrics = {
            f"summary/{prefix}/accuracy_matrix": builtin_matrix,
            f"summary/{prefix}/accuracy_matrix_labels": builtin_labels,
        }
        self.log_summary(summary_metrics)

        if not self.enabled or self.run is None:
            return

        table_columns = ["after_task"] + list(builtin_labels)
        table_data = []
        for task_idx, row in enumerate(builtin_matrix):
            table_data.append([task_idx] + list(row))
        matrix_table = self._wandb.Table(columns=table_columns, data=table_data)
        self._wandb.log({f"summary/{prefix}/accuracy_matrix_table": matrix_table})

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self._wandb.finish()
