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
    elif name == "spie_v4":
        from models.spie_v4 import Learner
    elif name == "spie_v5" or name == "spiev5":
        from models.spie_v5 import Learner
    elif name == "spie_v6" or name == "spiev6":
        from models.spie_v6 import Learner
    elif name == "spie_v7" or name == "spiev7":
        from models.spie_v7 import Learner
    elif name == "spie_v8" or name == "spiev8":
        from models.spie_v8 import Learner
    elif name == "spie":
        from models.spie import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    else:
        raise ValueError("Supported model names are 'tuna', 'tunamax', 'spie_v2', 'spie_v3', 'spie_v4', 'spie_v5', 'spie_v6', 'spie_v7', 'spie_v8', 'spie', and 'onlymax'.")
    return Learner(args)
