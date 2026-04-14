def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    elif name == "tunamax":
        from models.tunamax import Learner
    elif name == "spie_v13" or name == "spiev13":
        from models.spie_v13 import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    elif name == "min":
        from models.min import Learner
    elif name == "min_ablation":
        from models.min_ablation import Learner
    else:
        raise ValueError(
            "Supported model names are 'tuna', 'tunamax', 'spie_v13', "
            "'onlymax', 'min', and 'min_ablation'."
        )
    return Learner(args)
