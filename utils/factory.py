def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    elif name == "tunamax":
        from models.tunamax import Learner
    elif name == "spie_v2":
        from models.spie_v2 import Learner
    elif name == "spie_v3":
        from models.spie_v3 import Learner
    elif name == "spie":
        from models.spie import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    else:
        raise ValueError("Supported model names are 'tuna', 'tunamax', 'spie_v2', 'spie_v3', 'spie', and 'onlymax'.")
    return Learner(args)
