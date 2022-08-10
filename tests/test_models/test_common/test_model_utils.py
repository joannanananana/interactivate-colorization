# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
import torch.nn as nn

from mmdp.models.architectures.common import get_module_device
from mmdp.models.common import GANImageBuffer, set_requires_grad


def test_set_requires_grad():
    model = torch.nn.Conv2d(1, 3, 1, 1)
    set_requires_grad(model, False)
    for param in model.parameters():
        assert not param.requires_grad


def test_gan_image_buffer():
    # test buffer size = 0
    buffer = GANImageBuffer(buffer_size=0)
    img_np = np.random.randn(1, 3, 256, 256)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_return = buffer.query(img_tensor)
    assert torch.equal(img_tensor_return, img_tensor)

    # test buffer size > 0
    buffer = GANImageBuffer(buffer_size=1)
    img_np = np.random.randn(2, 3, 256, 256)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_0 = torch.unsqueeze(img_tensor[0], 0)
    img_tensor_1 = torch.unsqueeze(img_tensor[1], 0)
    img_tensor_00 = torch.cat([img_tensor_0, img_tensor_0], 0)
    img_tensor_return = buffer.query(img_tensor)
    assert (
        torch.equal(img_tensor_return, img_tensor)
        and torch.equal(buffer.image_buffer[0], img_tensor_0)
    ) or (
        torch.equal(img_tensor_return, img_tensor_00)
        and torch.equal(buffer.image_buffer[0], img_tensor_1)
    )

    # test buffer size > 0, specify buffer chance
    buffer = GANImageBuffer(buffer_size=1, buffer_ratio=0.3)
    img_np = np.random.randn(2, 3, 256, 256)
    img_tensor = torch.from_numpy(img_np)
    img_tensor_0 = torch.unsqueeze(img_tensor[0], 0)
    img_tensor_1 = torch.unsqueeze(img_tensor[1], 0)
    img_tensor_00 = torch.cat([img_tensor_0, img_tensor_0], 0)
    img_tensor_return = buffer.query(img_tensor)
    assert (
        torch.equal(img_tensor_return, img_tensor)
        and torch.equal(buffer.image_buffer[0], img_tensor_0)
    ) or (
        torch.equal(img_tensor_return, img_tensor_00)
        and torch.equal(buffer.image_buffer[0], img_tensor_1)
    )


def test_get_module_device_cpu():
    device = get_module_device(nn.Conv2d(3, 3, 3, 1, 1))
    assert device == torch.device("cpu")

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_get_module_device_cuda():
    module = nn.Conv2d(3, 3, 3, 1, 1).cuda()
    device = get_module_device(module)
    assert device == next(module.parameters()).get_device()

    # The input module should contain parameters.
    with pytest.raises(ValueError):
        get_module_device(nn.Flatten().cuda())
