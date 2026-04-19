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
    elif name == "spie_v15":
        from models.spie_v15 import Learner
    elif name == "spie_v16":
        from models.spie_v16 import Learner
    elif name == "spie_v18":
        from models.spie_v18 import Learner
    elif name == "spie_v17":
        from models.spie_v17 import Learner
    elif name == "ka_prompt":
        from models.ka_prompt import Learner
    elif name == "mqmk":
        from models.mqmk import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    elif name == "min":
        from models.min import Learner
    elif name == "min_ablation":
        from models.min_ablation import Learner
    elif name == "moal":
        from models.moal import Learner
    elif name == "mos":
        from models.mos import Learner
    elif name == "consistent_moe_prompt":
        from models.consistent_moe_prompt import Learner
    else:
        raise ValueError(
            "Supported model names are 'tuna', 'tunamax', 'spie_v13', 'spie_v14', 'spie_v15', 'spie_v16', 'spie_v17', 'spie_v18', "
            "'ka_prompt', 'mqmk', 'onlymax', 'min', 'min_ablation', 'moal', 'mos', and 'consistent_moe_prompt'."
        )
    return Learner(args)
