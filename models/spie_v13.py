from models.spie_v9 import Learner as SPiEV9Learner


class Learner(SPiEV9Learner):
    """SPiE v13 learner with shared LoRA and SVD-initialized expert VeRA bases."""

    _spie_version_name = "SPiE v13"
