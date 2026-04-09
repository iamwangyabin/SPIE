from models.spie_v9 import Learner as SPiEV9Learner


class Learner(SPiEV9Learner):
    """SPiE v11 learner with v9 training plus expert VeRA adapters."""

    _spie_version_name = "SPiE v11"
