from datetime import datetime
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
        self.enabled = bool(args.get("swanlab", False))
        self.run = None

        if not self.enabled:
            return


        import swanlab


        self._swanlab = swanlab
        config = _to_builtin(args)
        group = args.get("swanlab_group") or self._default_group(args)
        name = args.get("swanlab_experiment_name") or args.get("swanlab_name") or self._default_run_name(args)
        tags = args.get("swanlab_tags", [])
        project = args.get("swanlab_project") or self._default_project_name(args)

        init_kwargs = {
            "project": project,
            "config": config,
            "group": group,
            "experiment_name": name,
            "tags": tags,
            "mode": args.get("swanlab_mode", "online"),
            "description": args.get("swanlab_description"),
        }
        if args.get("swanlab_workspace"):
            init_kwargs["workspace"] = args["swanlab_workspace"]
        if args.get("swanlab_logdir"):
            init_kwargs["logdir"] = args["swanlab_logdir"]

        self.run = swanlab.init(**init_kwargs)

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
        note = str(args.get("note", "")).strip()
        if note:
            return "-".join([prefix, _sanitize_text(note)])
        return prefix

    def _default_project_name(self, args: Dict[str, Any]) -> str:
        prefix = _sanitize_text(args.get("prefix", "run"))
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"{prefix}-{date_str}"

    def log(self, metrics: Optional[Dict[str, Any]]) -> None:
        if not self.enabled or not metrics:
            return
        self._swanlab.log(_to_builtin(metrics))

    def log_train_history(self, task_id: int, train_history: Iterable[Dict[str, Any]]) -> None:
        for item in train_history:
            global_epoch = task_id * item["total_epochs"] + item["epoch"]
            payload = {}
            for key, value in item.items():
                if key in {"stage", "epoch", "total_epochs", "known_classes", "total_classes"}:
                    continue
                payload[f"train/{key}"] = value
            self._swanlab.log(_to_builtin(payload), step=global_epoch)

    def log_extra_history(self, task_id: int, extra_history: Iterable[Dict[str, Any]]) -> None:
        for item in extra_history:
            stage = item["stage"]
            total_epochs = item["total_epochs"]
            global_epoch = task_id * total_epochs + item["epoch"]
            payload = {}
            for key, value in item.items():
                if key in {"stage", "epoch", "total_epochs", "known_classes", "total_classes"}:
                    continue
                payload[f"extra/{stage}/{key}"] = value
            self._swanlab.log(_to_builtin(payload), step=global_epoch)

    def log_eval(
        self,
        cnn_accy: Dict[str, Any],
        nme_accy: Optional[Dict[str, Any]],
        step: int,
        avg_cnn: Optional[float] = None,
        avg_nme: Optional[float] = None,
    ) -> None:
        payload = {}
        payload.update(self._flatten_eval_metrics("cnn", cnn_accy))
        if avg_cnn is not None:
            payload["eval/cnn/avg_top1"] = avg_cnn

        if nme_accy is not None:
            payload.update(self._flatten_eval_metrics("nme", nme_accy))
            if avg_nme is not None:
                payload["eval/nme/avg_top1"] = avg_nme

        self._swanlab.log(_to_builtin(payload), step=step)

    def log_eval_variants(
        self,
        eval_variants: Optional[Dict[str, Dict[str, Any]]],
        step: int,
        avg_variants: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.enabled or not eval_variants:
            return

        payload = {}
        for name, accy in eval_variants.items():
            payload.update(self._flatten_eval_metrics(name, accy))
            if avg_variants and name in avg_variants:
                payload[f"eval/{name}/avg_top1"] = avg_variants[name]

        self._swanlab.log(_to_builtin(payload), step=step)

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
        if not self.enabled:
            return
        self._swanlab.log(_to_builtin(metrics))

    def log_accuracy_matrix(self, prefix: str, matrix, column_labels) -> None:
        if matrix is None or len(matrix) == 0:
            return

        builtin_matrix = _to_builtin(matrix)
        builtin_labels = _to_builtin(column_labels)
        if not self.enabled:
            return

        headers = ["after_task"] + list(builtin_labels)
        rows = []
        for task_idx, row in enumerate(builtin_matrix):
            formatted_row = [f"{float(value):.2f}" for value in row]
            rows.append([task_idx] + formatted_row)

        table = self._swanlab.echarts.Table()
        table.add(headers, rows)

        self._swanlab.log(
            {
                f"summary/{prefix}/accuracy_matrix": table
            }
        )

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self._swanlab.finish()
