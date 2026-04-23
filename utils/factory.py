def get_model(model_name, args):
    name = model_name.lower()
    if name == "aper":
        from models.aper import Learner
    elif name == "aper_finetune":
        from models.aper_finetune import Learner
    elif name == "aper_ssf":
        from models.aper_ssf import Learner
    elif name == "aper_vpt":
        from models.aper_vpt import Learner
    elif name == "aper_adapter":
        from models.aper_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "ease":
        from models.ease import Learner
    elif name == "slca":
        from models.slca import Learner
    elif name == "ranpac":
        from models.ranpac import Learner
    elif name == "fecam":
        from models.fecam import Learner
    elif name == "cofima":
        from models.cofima import Learner
    elif name == "tuna":
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
    elif name == "ssiat":
        from models.ssiat import Learner
    else:
        raise ValueError(
            "Supported model names are 'aper', 'aper_finetune', 'aper_ssf', "
            "'aper_vpt', 'aper_adapter', 'l2p', 'dualprompt', 'coda_prompt', 'ease', 'slca', "
            "'ranpac', 'fecam', 'cofima', 'tuna', 'tunamax', 'spie', 'ka_prompt', "
            "'mqmk', 'onlymax', 'min', 'min_ablation', 'moal', 'mos', 'ssiat', "
            "'consistent_moe_prompt', 'arcl', and 'vpt_nsp2pp'."
        )
    return Learner(args)
