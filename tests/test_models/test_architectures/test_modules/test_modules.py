# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

# InceptionV3代写
from mmdp.models.architectures.modules import (
    ColorizationResNet,
    InceptionV3,
    MidConvLayer,
    PostLayer,
    RRDBNet,
    UnetWideDecoder,
)


def test_rrdbnet_backbone():
    """Test RRDBNet backbone."""

    # model, initialization and forward (cpu)
    # x4 model
    net = RRDBNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        growth_channels=4,
        upscale_factor=4,
    )
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 12, 12)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 48, 48)

    # x2 model
    net = RRDBNet(
        in_channels=3,
        out_channels=3,
        mid_channels=8,
        num_blocks=2,
        growth_channels=4,
        upscale_factor=2,
    )
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 12, 12)
    img = _demo_inputs(input_shape)
    output = net(img)
    assert output.shape == (1, 3, 24, 24)

    # model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output = net(img.cuda())
        assert output.shape == (1, 3, 24, 24)

    with pytest.raises(TypeError):
        # pretrained should be str or None
        net.init_weights(pretrained=[1])


def _demo_inputs(input_shape=(1, 3, 64, 64)):
    """Create a superset of inputs needed to run backbone.

    Args:
        input_shape (tuple): input batch dimensions.
            Default: (1, 3, 64, 64).

    Returns:
        imgs: (Tensor): Images in FloatTensor with desired shapes.
    """
    imgs = np.random.random(input_shape)
    imgs = torch.FloatTensor(imgs)

    return imgs


class TestFIDInception:
    @classmethod
    def setup_class(cls):
        cls.load_fid_inception = False

    def test_fid_inception(self):
        inception = InceptionV3(load_fid_inception=self.load_fid_inception)
        imgs = torch.randn((2, 3, 256, 256))
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)

        imgs = torch.randn((2, 3, 512, 512))
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_fid_inception_cuda(self):
        inception = InceptionV3(load_fid_inception=self.load_fid_inception).cuda()
        imgs = torch.randn((2, 3, 256, 256)).cuda()
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)

        imgs = torch.randn((2, 3, 512, 512)).cuda()
        out = inception(imgs)[0]
        assert out.shape == (2, 2048, 1, 1)


def test_midconvlayer():
    mid = MidConvLayer(base_channels=3)
    imgs = torch.randn((2, 3, 256, 256))
    out = mid(imgs)
    assert out.shape == (2, 3, 256, 256)

    imgs = torch.randn((2, 3, 512, 512))
    out = mid(imgs)
    assert out.shape == (2, 3, 512, 512)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_midconvlayer_cuda():
    mid = MidConvLayer(base_channels=3).cuda()
    imgs = torch.randn((2, 3, 256, 256)).cuda()
    out = mid(imgs)
    assert out.shape == (2, 3, 256, 256)

    imgs = torch.randn((2, 3, 512, 512)).cuda()
    out = mid(imgs)
    assert out.shape == (2, 3, 512, 512)


def test_colorizationresnet():
    col = ColorizationResNet(num_layers=101)
    imgs = torch.randn((2, 3, 256, 256))
    out, _ = col(imgs)
    assert out.shape == (2, 2048, 8, 8)

    imgs = torch.randn((2, 3, 512, 512))
    out, _ = col(imgs)
    assert out.shape == (2, 2048, 16, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_colorizationresnet_cuda():
    col = ColorizationResNet(num_layers=101).cuda()
    imgs = torch.randn((2, 3, 256, 256)).cuda()
    out, short_cut = col(imgs)
    assert out.shape == (2, 2048, 8, 8)

    imgs = torch.randn((2, 3, 512, 512)).cuda()
    out, short_cut = col(imgs)
    short_cut_0 = short_cut[0].detach().cpu().numpy()
    short_cut_1 = short_cut[1].detach().cpu().numpy()
    short_cut_2 = short_cut[2].detach().cpu().numpy()
    short_cut_3 = short_cut[3].detach().cpu().numpy()
    assert out.shape == (2, 2048, 16, 16)
    assert short_cut_0.shape == (2, 64, 256, 256)
    assert short_cut_1.shape == (2, 256, 128, 128)
    assert short_cut_2.shape == (2, 512, 64, 64)
    assert short_cut_3.shape == (2, 1024, 32, 32)


def test_unetwidedecoder():
    col = ColorizationResNet(num_layers=101)
    mid = MidConvLayer(base_channels=2048)
    unetw = UnetWideDecoder(x_in_c_list=[64, 256, 512, 1024])

    imgs = torch.randn((2, 3, 256, 256))
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    out = unetw(res, short_cut_out)
    assert out.shape == (2, 256, 128, 128)

    imgs = torch.randn((2, 3, 512, 512))
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    out = unetw(res, short_cut_out)
    assert out.shape == (2, 256, 256, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_unetwidedecoder_cuda():
    col = ColorizationResNet(num_layers=101).cuda()
    mid = MidConvLayer(base_channels=2048).cuda()
    unetw = UnetWideDecoder(x_in_c_list=[64, 256, 512, 1024]).cuda()

    imgs = torch.randn((2, 3, 256, 256)).cuda()
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    out = unetw(res, short_cut_out)
    assert out.shape == (2, 256, 128, 128)

    imgs = torch.randn((2, 3, 512, 512)).cuda()
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    out = unetw(res, short_cut_out)
    assert out.shape == (2, 256, 256, 256)


def test_postlayer():
    col = ColorizationResNet(num_layers=101)
    mid = MidConvLayer(base_channels=2048)
    unetw = UnetWideDecoder(x_in_c_list=[64, 256, 512, 1024])
    postlayer = PostLayer()

    imgs = torch.randn((2, 3, 256, 256))
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    res = unetw(res, short_cut_out)
    out = postlayer(res, imgs)
    assert out.shape == (2, 3, 256, 256)

    imgs = torch.randn((2, 3, 512, 512))
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    res = unetw(res, short_cut_out)
    out = postlayer(res, imgs)
    assert out.shape == (2, 3, 512, 512)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_postlayer_cuda():
    col = ColorizationResNet(num_layers=101).cuda()
    mid = MidConvLayer(base_channels=2048).cuda()
    unetw = UnetWideDecoder(x_in_c_list=[64, 256, 512, 1024]).cuda()
    postlayer = PostLayer().cuda()

    imgs = torch.randn((2, 3, 256, 256)).cuda()
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    res = unetw(res, short_cut_out)
    out = postlayer(res, imgs)
    assert out.shape == (2, 3, 256, 256)

    imgs = torch.randn((2, 3, 512, 512)).cuda()
    res = imgs
    res, short_cut_out = col(res)
    res = mid(res)
    ## reverse the feature maps order in the list
    short_cut_out.reverse()
    res = unetw(res, short_cut_out)
    out = postlayer(res, imgs)
    assert out.shape == (2, 3, 512, 512)
