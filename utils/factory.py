def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    elif name == "tunamax":
        from models.tunamax import Learner
    elif name == "spie_v13":
        from models.spie_v13 import Learner
    elif name == "spie_v14":
        from models.spie_v14 import Learner
    elif name == "ka_prompt":
        from models.ka_prompt import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    elif name == "min":
        from models.min import Learner
    elif name == "min_ablation":
        from models.min_ablation import Learner
    elif name == "moal":
        from models.moal import Learner
    else:
        raise ValueError(
            "Supported model names are 'tuna', 'tunamax', 'spie_v13', 'spie_v14', "
            "'ka_prompt', 'onlymax', 'min', 'min_ablation', and 'moal'."
        )
    return Learner(args)
