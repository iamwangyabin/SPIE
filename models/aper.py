from copy import deepcopy


def Learner(args):
    routed_args = deepcopy(args)
    backbone_type = routed_args["backbone_type"].lower()

    if "_ssf" in backbone_type:
        from models.aper_ssf import Learner as RoutedLearner

        routed_args["model_name"] = "aper_ssf"
    elif "_vpt" in backbone_type:
        from models.aper_vpt import Learner as RoutedLearner

        routed_args["model_name"] = "aper_vpt"
    elif "_adapter" in backbone_type:
        from models.aper_adapter import Learner as RoutedLearner

        routed_args["model_name"] = "aper_adapter"
    else:
        from models.aper_finetune import Learner as RoutedLearner

        routed_args["model_name"] = "aper_finetune"

    return RoutedLearner(routed_args)
