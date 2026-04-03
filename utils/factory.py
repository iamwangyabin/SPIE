def get_model(model_name, args):
    name = model_name.lower()
    if name == "tuna":
        from models.tuna import Learner
    else:
        raise ValueError("Only 'tuna' is kept in this simplified project.")
    return Learner(args)
