import math
from torch import nn as nn
from .config import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg
from .layers import ConvBnAct, create_attn, SelectiveKernelConv_SKA
from .registry import register_model
from .resnet import ResNet

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'skresnext50_32x4d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnext50_ra-f40e40bf.pth'),
}

class SKABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, base_width=64, sk_kwargs=None, reduce_first=1, dilation=1, first_dilation=None,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, attn_layer=None, aa_layer=None,
                 drop_block=None, drop_path=None):
        super(SKABottleneck, self).__init__()

        sk_kwargs = sk_kwargs or {}
        conv_kwargs = dict(drop_block=drop_block, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = ConvBnAct(inplanes, first_planes, kernel_size=1, **conv_kwargs)
        self.conv2 = SelectiveKernelConv_SKA(
            planes,first_planes, width, stride=stride, dilation=first_dilation, groups=cardinality,
            **conv_kwargs, **sk_kwargs)
        conv_kwargs['act_layer'] = None
        self.conv3 = ConvBnAct(width, outplanes, kernel_size=1, **conv_kwargs)
        self.se = create_attn(attn_layer, outplanes)
        self.act = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        # x = self.attention(x,residual)
        x += residual
        x = self.act(x)
        return x

def _create_SKAresnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)


@register_model
def skresnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a Select Kernel ResNeXt50-32x4d model. This should be equivalent to
    the SKNet-50 model in the Select Kernel Paper
    """
    model_args = dict(
        block=SKABottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        zero_init_last_bn=False, **kwargs)
    return _create_SKAresnet('skresnext50_32x4d', pretrained, **model_args)
