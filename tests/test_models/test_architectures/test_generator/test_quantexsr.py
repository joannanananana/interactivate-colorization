# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdp.models.architectures.generator import FeMaSRNet


def test_quantexsr():
    """Test RRDBNet backbone."""

    # model, initialization and forward (cpu)
    # x4 model
    scale_factor = 4
    net = FeMaSRNet(in_channels=3, scale_factor=4, LQ_stage=True)
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 32, 32)
    img = _demo_inputs(input_shape)
    wsz = 8 // scale_factor * 8
    _, _, h_old, w_old = img.shape
    h_pad = (h_old // wsz + 1) * wsz - h_old
    w_pad = (w_old // wsz + 1) * wsz - w_old
    input = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
    input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, : w_old + w_pad]
    output, _, _, _, _ = net(input)
    output = output[..., : h_old * scale_factor, : w_old * scale_factor]
    assert output.shape == (1, 3, 128, 128)

    # x2 model
    scale_factor = 2
    net = FeMaSRNet(in_channels=3, scale_factor=4, LQ_stage=True)
    net.init_weights(pretrained=None)
    input_shape = (1, 3, 32, 32)
    img = _demo_inputs(input_shape)
    wsz = 8 // scale_factor * 8
    _, _, h_old, w_old = img.shape
    h_pad = (h_old // wsz + 1) * wsz - h_old
    w_pad = (w_old // wsz + 1) * wsz - w_old
    input = torch.cat([img, torch.flip(img, [2])], 2)[:, :, : h_old + h_pad, :]
    input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, : w_old + w_pad]
    output, _, _, _, _ = net(input)
    output = output[..., : h_old * scale_factor, : w_old * scale_factor]
    assert output.shape == (1, 3, 64, 64)

    # model forward (gpu)
    if torch.cuda.is_available():
        net = net.cuda()
        output, _, _, _, _ = net(input.cuda())
        output = output[..., : h_old * scale_factor, : w_old * scale_factor]
        assert output.shape == (1, 3, 64, 64)

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
