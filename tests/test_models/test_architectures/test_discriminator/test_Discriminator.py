# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdp.models.architectures.discriminator import (
    DeoldifyDiscriminator,
    UNetDiscriminatorSN,
)


def test_UNetDiscriminatorSN():
    # cpu
    disc = UNetDiscriminatorSN(num_in_ch=3)
    img = torch.randn(1, 3, 16, 16)
    disc(img)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        disc.init_weights(pretrained=233)

    # cuda
    if torch.cuda.is_available():
        disc = disc.cuda()
        img = img.cuda()
        disc(img)

        with pytest.raises(TypeError):
            # pretrained must be a string path
            disc.init_weights(pretrained=233)


def test_DeoldifyDiscriminator():
    # cpu
    disc = DeoldifyDiscriminator()
    img = torch.randn(1, 3, 256, 256)
    disc(img)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        disc.init_weights(pretrained=233)

    # cuda
    if torch.cuda.is_available():
        disc = disc.cuda()
        img = img.cuda()
        disc(img)

        with pytest.raises(TypeError):
            # pretrained must be a string path
            disc.init_weights(pretrained=233)
