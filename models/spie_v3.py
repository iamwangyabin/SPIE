from models.spie_v2 import Learner as SPiEV2Learner


class Learner(SPiEV2Learner):
    """SPiE v3 learner with expert-token-only full LoRA blocks."""

    def __init__(self, args):
        super().__init__(args)
        self.use_orth = False

    def orth_loss(self, features):
        del features
        return 0.0
