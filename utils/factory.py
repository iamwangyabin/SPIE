def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    elif name == "spie":
        from models.spie import Learner
    elif name == "onlymax":
        from models.onlymax import Learner
    else:
        raise ValueError("Supported model names are 'tuna', 'spie', and 'onlymax'.")
    return Learner(args)
