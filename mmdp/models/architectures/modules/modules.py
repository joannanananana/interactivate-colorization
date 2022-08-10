# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Optional, Tuple

import torch
from mmcv.cnn import kaiming_init, normal_init, xavier_init
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.weight_norm import weight_norm
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from mmdp.models.builder import MODULES


def conv1d(
    ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False
):
    """Create and initialize a `nn.Conv1d` layer with spectral
    normalization."""
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)


def batchnorm_2d(nf: int, norm_type: str = "NormBatch"):
    """A batchnorm2d layer with `nf` features initialized depending on
    `norm_type`."""
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0.0 if norm_type == "NormBatchZero" else 1.0)
    return bn


def sigmoid_range(x, low, high):
    """Sigmoid function with range `(low, high)`"""
    return torch.sigmoid(x) * (high - low) + low


def ifnone(a: Any, b: Any) -> Any:
    """`a` if `a` is not None, otherwise `b`."""
    return b if a is None else a


class MergeLayer(nn.Module):
    """Merge a shortcut with the result of the module by adding them or
    concatenating them if `dense=True`."""

    def __init__(self, dense: bool = False):
        super().__init__()
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


class SequentialEx(nn.Module):
    """Like `nn.Sequential`, but with ModuleList semantics, and can access
    module input."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for layer in self.layers:
            res.orig = x
            nres = layer(res)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, layer):
        return self.layers.append(layer)

    def extend(self, layer):
        return self.layers.extend(layer)

    def insert(self, i, layer):
        return self.layers.insert(i, layer)


def res_block(
    nf,
    dense: bool = False,
    norm_type: str = "NormBatch",
    bottle: bool = False,
    **conv_kwargs,
):
    """Resnet block of `nf` features.

    `conv_kwargs` are passed to `conv_layer`.
    """
    norm2 = norm_type
    if not dense and (norm_type == "NormBatch"):
        norm2 = "NormBatchZero"
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(
        conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
        MergeLayer(dense),
    )


class Flatten(nn.Module):
    """Flatten `x` to a single dimension, often used at the end of a model.

    `full` for rank-1 tensor
    """

    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class SelfAttention(nn.Module):
    """Self attention layer for nd."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x):
        # Notation from https://arxiv.org/pdf/1805.08318.pdf
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> nn.Module:
    """Initialize `m` weights with `func` and set `bias` to 0."""
    if func:
        if hasattr(m, "weight"):
            func(m.weight)
        if hasattr(m, "bias") and hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    return m


def relu(inplace: bool = False, leaky: float = None):
    """Return a relu activation, maybe `leaky` and `inplace`."""
    return (
        nn.LeakyReLU(inplace=inplace, negative_slope=leaky)
        if leaky is not None
        else nn.ReLU(inplace=inplace)
    )


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: str = "NormBatch",
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
    extra_bn: bool = False,
):
    """Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`)
    and batchnorm (if `bn`) layers."""
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ("NormBatch", "NormBatchZero") or extra_bn is True

    if bias is None:
        bias = not bn

    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d

    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )

    if norm_type == "NormWeight":
        conv = weight_norm(conv)
    elif norm_type == "NormSpectral":
        conv = spectral_norm(conv)

    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))

    return nn.Sequential(*layers)


def conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    padding: int = None,
    bias: bool = None,
    is_1d: bool = False,
    norm_type: str = "NormBatch",
    use_activ: bool = True,
    leaky: float = None,
    transpose: bool = False,
    init: Callable = nn.init.kaiming_normal_,
    self_attention: bool = False,
):
    """Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`)
    and batchnorm (if `bn`) layers."""
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in ("NormBatch", "NormBatchZero")
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(
        conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
        init,
    )
    if norm_type == "NormWeight":
        conv = weight_norm(conv)
    elif norm_type == "NormSpectral":
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """ICNR init of `x`, with `scale` and `init` function."""
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale**2))
    k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def conv_with_kaiming_uniform(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
):
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
    )
    # Caffe2 implementation uses XavierFill, which in fact
    # corresponds to kaiming_uniform_ in PyTorch
    torch.nn.init.kaiming_uniform_(conv.weight, a=1)
    torch.nn.init.constant_(conv.bias, 0)
    return conv


class PixelShuffle_ICNR(nn.Module):
    """Upsample by `scale` from `ni` filters to `nf` (default `ni`), using
    `nn.PixelShuffle`, `icnr` init, and `weight_norm`."""

    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        norm_type="Norm.Weight",
        leaky: float = None,
    ):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = conv_layer(
            ni, nf * (scale**2), ks=1, norm_type=norm_type, use_activ=False
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


class SigmoidRange(nn.Module):
    """Sigmoid module with range `(low,x_max)`"""

    def __init__(self, low, high):
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


class CustomPixelShuffle_ICNR(nn.Module):
    """Upsample by `scale` from `ni` filters to `nf` (default `ni`), using
    `nn.PixelShuffle`, `icnr` init, and `weight_norm`."""

    def __init__(
        self,
        ni: int,
        nf: int = None,
        scale: int = 2,
        blur: bool = False,
        leaky: float = None,
        **kwargs,
    ):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(
            ni, nf * (scale**2), ks=1, use_activ=False, **kwargs
        )
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x


def generation_init_weights(module, init_type="normal", init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal.
    """

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                normal_init(m, 0.0, init_gain)
            elif init_type == "xavier":
                xavier_init(m, gain=init_gain, distribution="normal")
            elif init_type == "kaiming":
                kaiming_init(
                    m,
                    a=0,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                    distribution="normal",
                )
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented"
                )
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


@MODULES.register_module()
class MidConvLayer(nn.Module):
    def __init__(
        self, norm_type: str = "NormSpectral", base_channels: int = 2048, **kwargs
    ):

        super().__init__()
        extra_bn = norm_type == "NormSpectral"

        kwargs_0 = {}
        middle_conv = nn.Sequential(
            custom_conv_layer(
                base_channels,
                base_channels * 2,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs_0,
            ),
            custom_conv_layer(
                base_channels * 2,
                base_channels,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs_0,
            ),
        ).eval()

        self.bn = batchnorm_2d(base_channels)
        self.relu = nn.ReLU()
        self.mid_cov = middle_conv

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.mid_cov(x)

        return x


@MODULES.register_module()
class ColorizationResNet(ResNet):
    def __init__(self, num_layers, pretrained=None, out_layers=[2, 5, 6, 7]):

        if num_layers == 101:
            super().__init__(
                block=Bottleneck,
                layers=[3, 4, 23, 3],
                num_classes=1,
                zero_init_residual=True,
            )
        elif num_layers == 34:
            super().__init__(
                block=BasicBlock,
                layers=[3, 4, 6, 3],
                num_classes=1,
                zero_init_residual=True,
            )
        else:
            raise NotImplementedError

        del self.avgpool
        del self.fc

        self.out_layers = out_layers

    def forward(self, x):
        layers = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        shortcut_out = []
        for layer_idx, layer in enumerate(layers):
            if layer_idx in self.out_layers:
                shortcut_out.append(x)
            x = layer(x)
        return x, shortcut_out

    def get_channels(self):
        x = torch.rand([1, 3, 64, 64])
        x = torch.Tensor(x)
        model_channels = []

        x = self.conv1(x)
        model_channels.append(x.shape[1])
        x = self.bn1(x)
        model_channels.append(x.shape[1])

        x = self.relu(x)
        model_channels.append(x.shape[1])
        x = self.maxpool(x)
        model_channels.append(x.shape[1])
        x = self.layer1(x)
        model_channels.append(x.shape[1])

        x = self.layer2(x)
        model_channels.append(x.shape[1])

        x = self.layer3(x)
        model_channels.append(x.shape[1])

        x = self.layer4(x)
        model_channels.append(x.shape[1])

        return model_channels


@MODULES.register_module()
class UnetBlockWide(nn.Module):
    """A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."""

    def __init__(
        self,
        up_in_c: int,
        x_in_c: int,
        n_out: int,
        final_div: bool = True,
        blur: bool = False,
        leaky: float = None,
        self_attention: bool = False,
        **kwargs,
    ):
        super().__init__()
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(
            up_in_c, up_out, blur=blur, leaky=leaky, **kwargs
        )
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(
            ni, x_out, leaky=leaky, self_attention=self_attention, **kwargs
        )
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor, s: Tensor) -> Tensor:
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


@MODULES.register_module()
class UnetWideDecoder(nn.Module):
    def __init__(
        self,
        self_attention: bool = True,
        x_in_c_list: list = [],
        ni: int = 2048,
        nf_factor: int = 2,
        blur: bool = True,
        norm_type: str = "NormSpectral",
    ):
        super().__init__()
        kwargs_0 = {}
        extra_bn = norm_type == "NormSpectral"

        layers_dec = []
        x_in_c_list.reverse()
        up_in_c = ni
        for i, x_in_c in enumerate(x_in_c_list):
            not_final = i != len(x_in_c_list) - 1
            sa = self_attention and (i == len(x_in_c_list) - 3)

            nf = 512 * nf_factor
            n_out = nf if not_final else nf // 2

            unet_block = UnetBlockWide(
                up_in_c,
                x_in_c,
                n_out,
                final_div=not_final,
                blur=blur,
                self_attention=sa,
                norm_type=norm_type,
                extra_bn=extra_bn,
                **kwargs_0,
            ).eval()
            up_in_c = n_out // 2
            layers_dec.append(unet_block)
        self.layers_dec = nn.ModuleList(layers_dec)

    def forward(self, x, short_cut_list):
        res = x
        for layer, s in zip(self.layers_dec, short_cut_list):
            res = layer(res, s)
        return res


@MODULES.register_module()
class PostLayer(nn.Module):
    def __init__(
        self,
        ni: int = 256,
        last_cross: bool = True,
        n_classes: int = 3,
        bottle: bool = False,
        norm_type: str = "NormSpectral",
        y_range: Optional[Tuple[float, float]] = (-3.0, 3.0),  # SigmoidRange
    ):
        super().__init__()
        kwargs_0 = {}
        layers_post = []
        layers_post.append(PixelShuffle_ICNR(ni, norm_type="NormWeight", **kwargs_0))
        if last_cross:
            layers_post.append(MergeLayer(dense=True))
            ni += 3
            layers_post.append(
                res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs_0)
            )
        layers_post += [
            custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)
        ]
        if y_range is not None:
            layers_post.append(SigmoidRange(*y_range))
        self.layers_post = nn.ModuleList(layers_post)

    def forward(self, x, x_short):
        res = x
        res = self.layers_post[0](res)
        res = torch.cat([res, x_short], dim=1)
        for idx, layer in enumerate(self.layers_post[2:]):
            res = layer(res)

        return res
