# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.runner import obj_from_dict

from mmdp.models import build_model
from mmdp.models.architectures.generator import SIGGRAPHGenerator
from mmdp.models.losses import GANLoss, L1Loss, L2Loss


def test_pix2pix():
    model_cfg = dict(
        type="Pix2Pix",
        generator=dict(type="SIGGRAPHGenerator", input_nc=4, output_nc=2),
        gan_loss=dict(type="GANLoss", gan_type="lsgan"),
        l1_loss=dict(type="L1Loss"),
        l2_loss=dict(type="L2Loss"),
        lambda_A=1,
    )
    ab_quant = 10  # quantization factor
    ab_max = 110  # maximimum ab value
    A = 2 * ab_max / ab_quant + 1
    B = A
    train_cfg = dict(ab_quant=10, ab_max=110, ab_norm=110, A=A, B=B, sample_p=0.125)
    test_cfg = dict(
        A=1,
        l_norm=100,
        l_cent=50,
        ab_norm=110,
    )
    # build pix2pixmodel
    pix2pixmodel = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert pix2pixmodel.__class__.__name__ == "Pix2Pix"
    assert isinstance(pix2pixmodel.generator, SIGGRAPHGenerator)
    assert isinstance(pix2pixmodel.l1_loss, L1Loss)
    assert isinstance(pix2pixmodel.gan_loss, GANLoss)
    assert isinstance(pix2pixmodel.l2_loss, L2Loss)

    # prepare data
    inputs = torch.rand(1, 3, 32, 32)
    data_batch = {"pair": inputs}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        "generator": obj_from_dict(
            optim_cfg,
            torch.optim,
            dict(params=getattr(pix2pixmodel, "generator").parameters()),
        ),
    }

    # no forward train in GAN models, raise ValueError
    # with pytest.raises(ValueError):
    #    pix2pixmodel(data_batch=data_batch, test_mode=False)
    with torch.no_grad():
        outputs = pix2pixmodel(data_batch=data_batch, test_mode=False)
    assert torch.is_tensor(outputs["fake_B_reg"])
    assert outputs["fake_B_reg"].size() == (1, 2, 32, 32)
    # test forward_test
    with torch.no_grad():
        outputs = pix2pixmodel(data_batch=data_batch, test_mode=True)
    assert torch.is_tensor(outputs["fake_b"])
    assert outputs["fake_b"].size() == (1, 2, 32, 32)

    # model&generator is not actually matched
    # forward of generator takes 3 but forward of model takes 1 input
    with pytest.raises(TypeError):
        _, _ = pix2pixmodel.forward_dummy(torch.rand(1, 32, 32, 3) * 255)

    # val_step
    with torch.no_grad():
        _, outputs = pix2pixmodel.val_step(inputs)
    assert torch.is_tensor(outputs)
    assert outputs.size() == (1, 2, 32, 32)

    # test train_step
    outputs = pix2pixmodel.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)

    for v in ["loss_g", "G_entr"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    assert torch.is_tensor(outputs["results"]["fake_b"])
    assert outputs["results"]["fake_b"].size() == (1, 2, 32, 32)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        pix2pixmodel = pix2pixmodel.cuda()
        optimizer = {
            "generator": obj_from_dict(
                optim_cfg,
                torch.optim,
                dict(params=getattr(pix2pixmodel, "generator").parameters()),
            ),
        }
        data_batch = {
            "pair": inputs.cuda(),
        }

        # forward_test
        with torch.no_grad():
            outputs = pix2pixmodel(data_batch=data_batch, test_mode=True)
        assert torch.is_tensor(outputs["fake_b"])
        assert outputs["fake_b"].size() == (1, 2, 32, 32)

        # val_step
        with torch.no_grad():
            _, outputs = pix2pixmodel.val_step(inputs.cuda())
        assert torch.is_tensor(outputs)
        assert outputs.size() == (1, 2, 32, 32)

        # train_step
        outputs = pix2pixmodel.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)

        for v in ["loss_g", "G_entr"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.is_tensor(outputs["results"]["fake_b"])
        assert outputs["results"]["fake_b"].size() == (1, 2, 32, 32)

    # test parameters in train_cfg
    with pytest.raises(RuntimeError):
        data_batch = {"pair": inputs.cpu()}
        train_cfg_ = dict(ab_quant=10, ab_max=10, ab_norm=10, sample_p=0.125, A=A, B=B)
        pix2pixmodel = build_model(model_cfg, train_cfg=train_cfg_, test_cfg=test_cfg)
        outputs = pix2pixmodel.train_step(data_batch, optimizer)

    # test no discriminator (testing mode)
    model_cfg_ = model_cfg.copy()
    pix2pixmodel = build_model(model_cfg_, train_cfg=train_cfg, test_cfg=test_cfg)
    with torch.no_grad():
        outputs = pix2pixmodel(data_batch=data_batch, test_mode=True)
    assert torch.is_tensor(outputs["fake_b"])
    assert outputs["fake_b"].size() == (1, 2, 32, 32)

    # test without pixel loss and perceptual loss
    with pytest.raises(TypeError):
        model_cfg_ = model_cfg.copy()
        model_cfg_.pop("gan_loss")
        model_cfg_.pop("l1_loss")
        model_cfg_.pop("l2_loss")
        pix2pixmodel = build_model(model_cfg_, train_cfg=None, test_cfg=None)

    # test train_step w/o pixel_loss
    model_cfg["pixel_loss"] = dict(type="L1Loss", loss_weight=1.0, reduction="mean")
    pix2pixmodel = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    data_batch = {"pair": inputs}
    outputs = pix2pixmodel.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_g", "G_entr"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    assert torch.is_tensor(outputs["results"]["fake_b"])
    assert outputs["results"]["fake_b"].size() == (1, 2, 32, 32)
