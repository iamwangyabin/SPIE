from models.spie_v8 import Learner as SPiEV8Learner


class Learner(SPiEV8Learner):
    """SPiE v10 learner with MLP-output-only shared/expert LoRA branches."""

    _spie_version_name = "SPiE v10"
