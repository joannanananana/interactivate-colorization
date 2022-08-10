import pytest
import torch
from mmcv import Config

from mmdp.models import build_model


def test_psnr():
    opt = Config()
    opt.model = dict(
        type="Pix2Pix",
        generator=dict(
            type="SIGGRAPHGenerator", input_nc=4, output_nc=2, classification=True
        ),
        gan_loss=dict(type="GANLoss", gan_type="lsgan"),
        l1_loss=dict(type="L1Loss"),
        l2_loss=dict(type="L2Loss"),
        lambda_A=1,
    )
    ab_quant = 10.0  # quantization factor
    ab_max = 110.0  # maximimum ab value
    A = 2 * ab_max / ab_quant + 1
    B = A
    opt.train_cfg = dict(
        ab_quant=10.0, ab_max=110.0, ab_norm=110.0, A=A, B=B, sample_p=1.0
    )
    opt.test_cfg = dict(
        A=1.0,
        l_norm=100.0,
        l_cent=50,
        ab_norm=110,
    )
    model = build_model(opt.model, train_cfg=opt.train_cfg, test_cfg=opt.test_cfg)
    with pytest.raises(AttributeError):
        test_data = torch.randn(1, 4, 176, 176)  # wrong size of input
        result = model.forward(test_data, psnr=True)
        print(result)
    with pytest.raises(AttributeError):
        visuals = model.get_current_visuals()
    model.real_a = torch.randn(1, 2, 16, 16)
    model.real_B = torch.randn(1, 2, 16, 16)
    model.fake_B_reg = torch.randn(1, 2, 16, 16)
    visuals = model.get_current_visuals()
    assert isinstance(visuals, dict)
    assert visuals["fake_reg"].size() == (1, 3, 16, 16)
