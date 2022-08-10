# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.runner import obj_from_dict

from mmdp.models import build_model
from mmdp.models.architectures.discriminator import UNetDiscriminatorSN
from mmdp.models.architectures.generator import FeMaSRNet
from mmdp.models.losses import CodebookLoss, GANLoss, L1Loss, PerceptualLoss


def test_Quantexsrganmodel():

    model_cfg = dict(
        type="QuanTexSRGANModel",
        generator=dict(
            type="FeMaSRNet",
            in_channel=3,
            beta=0.25,
            scale_factor=4,
            gt_resolution=256,
            norm_type="gn",
            act_type="silu",
            frozen_module_keywords=["quantize"],
            LQ_stage=True,
        ),
        discriminator=dict(
            type="UNetDiscriminatorSN", num_in_ch=3, skip_connection=True
        ),
        pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
        perceptual_loss=dict(
            type="PerceptualLoss",
            layer_weights={
                "2": 0.1,
                "7": 0.1,
                "16": 1.0,
                "25": 1.0,
                "34": 1.0,
            },
            vgg_type="vgg19",
            perceptual_weight=1.0,
            style_weight=100,
            norm_img=False,
        ),
        gan_loss=dict(
            type="GANLoss",
            gan_type="hinge",
            loss_weight=0.1,
            real_label_val=1.0,
            fake_label_val=0,
        ),
        ####!!!需要写codebookloss###
        codebook_loss=dict(type="CodebookLoss", loss_weight=1),
        is_use_sharpened_gt_in_pixel=False,
        is_use_sharpened_gt_in_percep=False,
        is_use_sharpened_gt_in_gan=False,
        is_use_ema=False,
    )

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == "QuanTexSRGANModel"
    assert isinstance(restorer.generator, FeMaSRNet)
    assert isinstance(restorer.discriminator, UNetDiscriminatorSN)
    assert isinstance(restorer.pixel_loss, L1Loss)
    assert isinstance(restorer.gan_loss, GANLoss)
    assert isinstance(restorer.perceptual_loss, PerceptualLoss)
    assert isinstance(restorer.codebook_loss, CodebookLoss)

    # prepare data
    inputs = torch.rand(1, 3, 32, 32)
    targets = torch.rand(1, 3, 128, 128)
    data_batch = {"lq": inputs, "gt": targets, "gt_unsharp": targets}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        "generator": obj_from_dict(
            optim_cfg,
            torch.optim,
            dict(params=getattr(restorer, "generator").parameters()),
        ),
        "discriminator": obj_from_dict(
            optim_cfg,
            torch.optim,
            dict(params=getattr(restorer, "discriminator").parameters()),
        ),
    }

    # no forward train in GAN models, raise ValueError
    with pytest.raises(ValueError):
        restorer(**data_batch, test_mode=False)

    # test forward_test
    data_batch.pop("gt_unsharp")
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    # assert torch.equal(outputs["lq"], data_batch["lq"])
    assert torch.is_tensor(outputs["output"])
    assert outputs["output"].size() == (1, 3, 128, 128)

    # test forward_dummy
    with torch.no_grad():
        output, _, _, _, _ = restorer.forward_dummy(data_batch["lq"])
    assert torch.is_tensor(output)
    assert output.size() == (1, 3, 128, 128)

    # val_step
    with torch.no_grad():
        outputs = restorer.val_step(data_batch)
    data_batch["gt_unsharp"] = targets
    # assert torch.equal(outputs["lq"], data_batch["lq"])
    assert torch.is_tensor(outputs["output"])
    assert outputs["output"].size() == (1, 3, 128, 128)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_perceptual", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    # assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
    assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
    assert torch.is_tensor(outputs["results"]["output"])
    assert outputs["results"]["output"].size() == (1, 3, 128, 128)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        optimizer = {
            "generator": obj_from_dict(
                optim_cfg,
                torch.optim,
                dict(params=getattr(restorer, "generator").parameters()),
            ),
            "discriminator": obj_from_dict(
                optim_cfg,
                torch.optim,
                dict(params=getattr(restorer, "discriminator").parameters()),
            ),
        }
        data_batch = {
            "lq": inputs.cuda(),
            "gt": targets.cuda(),
            "gt_unsharp": targets.cuda(),
        }

        # forward_test
        data_batch.pop("gt_unsharp")
        with torch.no_grad():
            outputs = restorer(**data_batch, test_mode=True)
        # assert torch.equal(outputs["lq"], data_batch["lq"].cpu())
        assert torch.is_tensor(outputs["output"])
        assert outputs["output"].size() == (1, 3, 128, 128)

        # val_step
        with torch.no_grad():
            outputs = restorer.val_step(data_batch)
        data_batch["gt_unsharp"] = targets.cuda()
        # assert torch.equal(outputs["lq"], data_batch["lq"].cpu())
        assert torch.is_tensor(outputs["output"])
        assert outputs["output"].size() == (1, 3, 128, 128)

        # train_step
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        for v in [
            "loss_perceptual",
            "loss_gan",
            "loss_d_real",
            "loss_d_fake",
            "loss_pix",
        ]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        # assert torch.equal(outputs["results"]["lq"], data_batch["lq"].cpu())
        assert torch.equal(outputs["results"]["gt"], data_batch["gt"].cpu())
        assert torch.is_tensor(outputs["results"]["output"])
        assert outputs["results"]["output"].size() == (1, 3, 128, 128)

    # test disc_steps and disc_init_steps and start_iter
    data_batch = {"lq": inputs.cpu(), "gt": targets.cpu(), "gt_unsharp": targets.cpu()}
    train_cfg = dict(disc_steps=2, disc_init_steps=2, start_iter=0)
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_d_real", "loss_d_fake"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    # assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
    assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
    assert torch.is_tensor(outputs["results"]["output"])
    assert outputs["results"]["output"].size() == (1, 3, 128, 128)

    # test no discriminator (testing mode)
    model_cfg_ = model_cfg.copy()
    model_cfg_.pop("discriminator")
    restorer = build_model(model_cfg_, train_cfg=train_cfg, test_cfg=test_cfg)
    data_batch.pop("gt_unsharp")
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    data_batch["gt_unsharp"] = targets.cpu()
    # assert torch.equal(outputs["lq"], data_batch["lq"])
    assert torch.is_tensor(outputs["output"])
    assert outputs["output"].size() == (1, 3, 128, 128)

    # test without pixel loss and perceptual loss
    model_cfg_ = model_cfg.copy()
    model_cfg_.pop("pixel_loss")
    restorer = build_model(model_cfg_, train_cfg=None, test_cfg=None)

    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_gan", "loss_d_real", "loss_d_fake"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    # assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
    assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
    assert torch.is_tensor(outputs["results"]["output"])
    assert outputs["results"]["output"].size() == (1, 3, 128, 128)

    # test train_step w/o loss_percep
    restorer = build_model(model_cfg, train_cfg=None, test_cfg=None)

    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_style", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    # assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
    assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
    assert torch.is_tensor(outputs["results"]["output"])
    assert outputs["results"]["output"].size() == (1, 3, 128, 128)

    # test train_step w/o loss_style
    restorer = build_model(model_cfg, train_cfg=None, test_cfg=None)

    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_perceptual", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    # assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
    assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
    assert torch.is_tensor(outputs["results"]["output"])
    assert outputs["results"]["output"].size() == (1, 3, 128, 128)
