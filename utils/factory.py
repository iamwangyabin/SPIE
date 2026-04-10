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
    elif name == "spie_v10" or name == "spiev10":
        from models.spie_v10 import Learner
    elif name == "spie_v12" or name == "spiev12":
        from models.spie_v12 import Learner
    elif name == "spie_v13" or name == "spiev13":
        from models.spie_v13 import Learner
    elif name == "spie_v11" or name == "spiev11":
        from models.spie_v11 import Learner
    elif name == "spie_v9" or name == "spiev9":
        from models.spie_v9 import Learner
    elif name == "spie":
        from models.spie import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    else:
        raise ValueError("Supported model names are 'tuna', 'tunamax', 'spie_v2', 'spie_v3', 'spie_v4', 'spie_v5', 'spie_v6', 'spie_v7', 'spie_v8', 'spie_v9', 'spie_v10', 'spie_v11', 'spie_v12', 'spie_v13', 'spie', and 'onlymax'.")
    return Learner(args)
