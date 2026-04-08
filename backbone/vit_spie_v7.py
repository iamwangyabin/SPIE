from backbone.vit_spie_v6 import VisionTransformer
from backbone.vit_spie_v6 import vit_base_patch16_224_in21k_spie_v6
from backbone.vit_spie_v6 import vit_base_patch16_224_spie_v6


def vit_base_patch16_224_spie_v7(pretrained=False, **kwargs):
    return vit_base_patch16_224_spie_v6(pretrained=pretrained, **kwargs)


def vit_base_patch16_224_in21k_spie_v7(pretrained=False, **kwargs):
    return vit_base_patch16_224_in21k_spie_v6(pretrained=pretrained, **kwargs)


vit_base_patch16_224_spiev7 = vit_base_patch16_224_spie_v7
vit_base_patch16_224_in21k_spiev7 = vit_base_patch16_224_in21k_spie_v7
