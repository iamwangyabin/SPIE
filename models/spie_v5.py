from models.spie_v2 import Learner as SPiEV2Learner


class Learner(SPiEV2Learner):
    """TunaMax learner with SPiE v5 full-sequence MLP-LoRA adapters."""

    def __init__(self, args):
        super().__init__(args)
        self.use_orth = False

    def _should_reset_task_modules(self):
        return self._cur_task >= 0

    def orth_loss(self, features):
        del features
        return 0.0
