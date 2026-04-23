def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    elif name == "tunamax":
        from models.tunamax import Learner
    elif name == "spie":
        from models.spie import Learner
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
    elif name == "arcl":
        from models.arcl import Learner
    elif name == "vpt_nsp2pp":
        from models.vpt_nsp2pp import Learner
    else:
        raise ValueError(
            "Supported model names are 'tuna', 'tunamax', 'spie', "
            "'ka_prompt', 'mqmk', 'onlymax', 'min', 'min_ablation', 'moal', 'mos', 'consistent_moe_prompt', 'arcl', and 'vpt_nsp2pp'."
        )
    return Learner(args)
