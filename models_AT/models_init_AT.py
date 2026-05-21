from torch import nn
from IFKD.models_AT.ResNet.network import ResNetAE
from IFKD.models_AT.SwinJSCC.network import SwinJSCC_AT
from IFKD.models_AT.Conv.network import ConvJSCC_AT
from IFKD.models_AT.VitJSCC.network import VitJSCC_AT


def resnet_AT_C128(**kwargs):
    model = ResNetAE(C=128, embed_dims=[64, 128, 192, 256], num_blocks=[2, 2, 2, 2], **kwargs)
    return model

def resnet_AT_C96(**kwargs):
    model = ResNetAE(C=96, embed_dims=[64, 128, 192, 256], num_blocks=[2, 2, 2, 2], **kwargs)
    return model

def resnet_AT_C192(**kwargs):
    model = ResNetAE(C=192, embed_dims=[64, 128, 192, 256], num_blocks=[2, 2, 2, 2], **kwargs)
    return model

def resnet_AT_C256(**kwargs):
    model = ResNetAE(C=256, embed_dims=[64, 128, 192, 256], num_blocks=[2, 2, 2, 2], **kwargs)
    return model

def resnet_AT_C320(**kwargs):
    model = ResNetAE(C=320, embed_dims=[64, 128, 192, 256], num_blocks=[2, 2, 2, 2], **kwargs)
    return model

def swinjscc_AT_C128(**kwargs):
    model = SwinJSCC_AT(
        encoder_kwargs=dict(
            img_size=(256, 256), patch_size=2, in_chans=3,
            embed_dims=[64, 96, 128, 160], depths=[2, 2, 2, 2], num_heads=[2, 3, 4, 5],
            C=128, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        ),
        decoder_kwargs=dict(
            img_size=(256, 256),
            embed_dims=[160, 128, 96, 64], depths=[2, 2, 2, 2], num_heads=[5, 4, 3, 2],
            C=128, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        ),
        downsample=4,
        **kwargs)
    return model


def conv_AT_C128(**kwargs):
    model = ConvJSCC_AT(C=128, encoder_embed_dims=[64, 128, 192, 256], encoder_num_blocks=[2, 2, 2, 2],
                     decoder_embed_dims=[256, 192, 128, 64], decoder_num_blocks=[2, 2, 2, 2], **kwargs)
    return model


def vitjscc_AT_C128(**kwargs):
    model = VitJSCC_AT(img_size=256, C=128, patch_size=16,
                       encoder_num_blocks=[2, 2, 2, 2],
                       decoder_num_blocks=[2, 2, 2, 2],
                       embed_dim=256,
                       num_head=8,
                       **kwargs)
    return model
