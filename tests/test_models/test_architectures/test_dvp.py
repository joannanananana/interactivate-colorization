import pytest
import torch

from mmdp.models.architectures.generator.vgg_arch import VGGFeatureExtractor as vgg19

# import sys,os
# sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from mmdp.models.architectures.modules.generator.Unet_net import UnetGenerator as UNet


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_unet():
    img = torch.randn((1, 3, 512, 512)).to("cuda")
    net = UNet(in_channels=3, out_channels=3, base_channels=32, kernel3=True).to("cuda")
    out = net(img)
    assert out.shape == img.shape
    assert torch.is_tensor(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_vgg19_cuda():
    in_tensor = torch.randn((1, 3, 512, 512)).float().to("cuda")
    layer_name_list = {"relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"}
    VGG_19 = vgg19(layer_name_list).to("cuda")
    out = VGG_19(in_tensor)
    assert isinstance(out, dict)
