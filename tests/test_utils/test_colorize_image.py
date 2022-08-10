import pytest
import torch

from mmdp.utils.colorize_image import ColorizeImageTorch


def test_ColorizeImageTorch():
    colorModel = ColorizeImageTorch(Xd=256)
    data = torch.randn(1, 3)
    with pytest.raises(TypeError):
        colorModel.net_forward(data) == -1
    data2 = torch.randn(1, 3)
    assert colorModel.net_forward(data, data2) == -1
    with pytest.raises(FileNotFoundError):
        colorModel.prep_net()
