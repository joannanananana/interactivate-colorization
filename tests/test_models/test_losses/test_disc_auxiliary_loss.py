# Copyright (c) OpenMMLab. All rights reserved.
import numpy.testing as npt
import torch

from mmdp.models import DiscShiftLoss, GradientPenaltyLoss


def test_disc_shift_loss():
    loss_disc_shift = DiscShiftLoss()
    x = torch.Tensor([0.1])
    loss = loss_disc_shift(x)

    npt.assert_almost_equal(loss.item(), 0.01)


def test_gradient_penalty_losses():
    """Test gradient penalty losses."""
    input = torch.ones(1, 3, 6, 6) * 2

    gan_loss = GradientPenaltyLoss(loss_weight=10.0)
    loss = gan_loss(lambda x: x, input, input, mask=None)
    assert loss.item() > 0
    mask = torch.ones(1, 3, 6, 6)
    mask[:, :, 2:4, 2:4] = 0
    loss = gan_loss(lambda x: x, input, input, mask=mask)
    assert loss.item() > 0
