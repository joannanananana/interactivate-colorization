import numpy as np
import pytest
import torch

from mmdp.models.architectures.generator import SIGGRAPHGenerator


def test_SIGGRAPHGenerator():
    # test output.shape
    net = SIGGRAPHGenerator(input_nc=4, output_nc=2)
    net.init_weights(pretrained=None)
    input_A = (1, 2, 16, 16)
    input_B = (1, 1, 16, 16)
    mask_B = (1, 1, 16, 16)
    img_A = _demo_inputs(input_A)
    img_B = _demo_inputs(input_B)
    msk_B = _demo_inputs(mask_B)

    _, regr = net(img_A, img_B, msk_B)
    assert regr.shape == (1, 2, 16, 16)

    if torch.cuda.is_available():
        net.cuda()
        _, regr = net(img_A.cuda(), img_B.cuda(), msk_B.cuda())
        assert regr.shape == (1, 2, 16, 16)
        net.cpu()

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
