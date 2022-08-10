import torch
from mmcv.runner import obj_from_dict

from mmdp.models import build_model
from mmdp.models.architectures import DeoldifyDiscriminator, DeOldifyGenerator
from mmdp.models.losses import GANLoss, PerceptualLoss


def test_deoldify_donly():
    model_cfg = dict(
        type="DeOldify",
        generator=dict(
            type="DeOldifyGenerator",
            encoder=dict(
                type="ColorizationResNet",
                num_layers=101,
                pretrained=None,
                out_layers=[2, 5, 6, 7],
            ),
            mid_layers=dict(
                type="MidConvLayer", norm_type="NormSpectral", base_channels=2048
            ),
            decoder=dict(
                type="UnetWideDecoder",
                self_attention=True,
                x_in_c_list=[64, 256, 512, 1024],
                ni=2048,
                nf_factor=2,
                norm_type="NormSpectral",
            ),
            post_layers=dict(
                type="PostLayer",
                ni=256,
                last_cross=True,
                n_classes=3,
                bottle=False,
                norm_type="NormSpectral",
                y_range=(-3.0, 3.0),
            ),
        ),
        discriminator=dict(
            type="DeoldifyDiscriminator",
            in_channels=6,
            base_channels=256,
            num_blocks=3,
            p=0.15,
        ),
        nogan_stage="Donly",
        gan_loss=dict(
            type="GANLoss",
            gan_type="vanilla",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.0,
        ),
        perceptual_loss=dict(
            type="PerceptualLoss",
            layer_weights={
                "18": 20.0,
                "27": 70.0,
                "36": 10.0,
            },
            vgg_type="vgg19",
            perceptual_weight=1.5,
            style_weight=0,
            criterion="mse",
        ),
        l1_loss=dict(loss_weight=1.0),
    )

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    # test attributes
    assert restorer.__class__.__name__ == "DeOldify"
    assert isinstance(restorer.generator, DeOldifyGenerator)
    assert isinstance(restorer.discriminator, DeoldifyDiscriminator)
    assert isinstance(restorer.gan_loss, GANLoss)
    assert isinstance(restorer.perceptual_loss, PerceptualLoss)

    # prepare data
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 3, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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

    # test forward_test
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # test forward_train
    outputs = restorer(**data_batch, test_mode=False)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"])
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    assert isinstance(outputs["results"], dict)
    for v in ["loss_gan_d_fake", "loss_gan_d_real"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1

    assert torch.equal(
        outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
    )
    assert torch.equal(
        outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
    )
    assert torch.is_tensor(outputs["results"]["image_color_fake"])
    assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)

    # test disc_steps and disc_init_steps
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 3, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    test_cfg = dict()

    # build restorer
    # model_cfg["generator"]["decoder"]["x_in_c_list"]=[64, 256, 512, 1024]
    # 这个道理告诉我们，善用reverse,因为用了一次reverse导致数组翻转
    model_cfg["generator"]["decoder"]["x_in_c_list"] = [64, 256, 512, 1024]
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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
    # iter 0, 1
    for i in range(2):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        for v in ["loss_gan_d_fake", "loss_gan_d_real"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)
        # assert restorer.iteration == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        # assert restorer.iteration == i
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        log_check_list = ["loss_gan_d_fake", "loss_gan_d_real"]
        for v in log_check_list:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)
        # assert restorer.iteration == i + 1


def test_deoldify_gandD():
    model_cfg = dict(
        type="DeOldify",
        generator=dict(
            type="DeOldifyGenerator",
            encoder=dict(
                type="ColorizationResNet",
                num_layers=101,
                pretrained=None,
                out_layers=[2, 5, 6, 7],
            ),
            mid_layers=dict(
                type="MidConvLayer", norm_type="NormSpectral", base_channels=2048
            ),
            decoder=dict(
                type="UnetWideDecoder",
                self_attention=True,
                x_in_c_list=[64, 256, 512, 1024],
                ni=2048,
                nf_factor=2,
                norm_type="NormSpectral",
            ),
            post_layers=dict(
                type="PostLayer",
                ni=256,
                last_cross=True,
                n_classes=3,
                bottle=False,
                norm_type="NormSpectral",
                y_range=(-3.0, 3.0),
            ),
        ),
        discriminator=dict(
            type="DeoldifyDiscriminator",
            in_channels=6,
            base_channels=256,
            num_blocks=3,
            p=0.15,
        ),
        nogan_stage="GandD",
        gan_loss=dict(
            type="GANLoss",
            gan_type="vanilla",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0,
        ),
        perceptual_loss=dict(
            type="PerceptualLoss",
            layer_weights={
                "18": 20.0,
                "27": 70.0,
                "36": 10.0,
            },
            vgg_type="vgg19",
            perceptual_weight=1.5,
            style_weight=0,
            criterion="mse",
        ),
        l1_loss=dict(loss_weight=1.0),
    )
    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    # test attributes
    assert restorer.__class__.__name__ == "DeOldify"
    assert isinstance(restorer.generator, DeOldifyGenerator)
    assert isinstance(restorer.discriminator, DeoldifyDiscriminator)
    assert isinstance(restorer.gan_loss, GANLoss)
    assert isinstance(restorer.perceptual_loss, PerceptualLoss)

    # prepare data
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 3, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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

    # test forward_test
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # test forward_train
    outputs = restorer(**data_batch, test_mode=False)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"])
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # val_step
    with torch.no_grad():
        outputs = restorer.val_step(data_batch)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    assert isinstance(outputs["results"], dict)
    for v in [
        "loss_gan_d_fake",
        "loss_gan_d_real",
        "loss_gan_g",
        "loss_perceptual",
        "l1_loss",
    ]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1

    assert torch.equal(
        outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
    )
    assert torch.equal(
        outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
    )
    assert torch.is_tensor(outputs["results"]["image_color_fake"])
    assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)

    # test disc_steps and disc_init_steps
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 3, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    test_cfg = dict()

    # build restorer
    # model_cfg["generator"]["decoder"]["x_in_c_list"]=[64, 256, 512, 1024]
    # 这个道理告诉我们，善用reverse,因为用了一次reverse导致数组翻转
    model_cfg["generator"]["decoder"]["x_in_c_list"] = [64, 256, 512, 1024]
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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
    # iter 0, 1
    for i in range(2):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        for v in ["loss_gan_d_fake", "loss_gan_d_real"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)
        # assert restorer.iteration == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        # assert restorer.iteration == i
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        log_check_list = ["loss_gan_d_fake", "loss_gan_d_real"]
        for v in log_check_list:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)
        # assert restorer.iteration == i + 1


def test_deoldify_gandD_Lab():
    model_cfg = dict(
        type="DeOldify",
        use_lab=True,
        generator=dict(
            type="DeOldifyGenerator",
            encoder=dict(
                type="ColorizationResNet",
                num_layers=101,
                pretrained=None,
                out_layers=[2, 5, 6, 7],
            ),
            mid_layers=dict(
                type="MidConvLayer", norm_type="NormSpectral", base_channels=2048
            ),
            decoder=dict(
                type="UnetWideDecoder",
                self_attention=True,
                x_in_c_list=[64, 256, 512, 1024],
                ni=2048,
                nf_factor=2,
                norm_type="NormSpectral",
            ),
            post_layers=dict(
                type="PostLayer",
                ni=256,
                last_cross=True,
                n_classes=2,
                bottle=False,
                norm_type="NormSpectral",
                y_range=(-3.0, 3.0),
            ),
        ),
        discriminator=dict(
            type="DeoldifyDiscriminator",
            in_channels=3,  # 原configs是3
            base_channels=256,
            num_blocks=3,
            p=0.15,
        ),
        nogan_stage="GandD",
        gan_loss=dict(
            type="GANLoss",
            gan_type="vanilla",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=1.0,
        ),
        perceptual_loss=dict(
            type="PerceptualLoss",
            layer_weights={
                "18": 20.0,
                "27": 70.0,
                "36": 10.0,
            },
            use_input_norm=False,
            vgg_type="vgg19",
            perceptual_weight=1.5,
            style_weight=0,
            criterion="mse",
        ),
        l1_loss=dict(loss_weight=1.0),
    )
    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    # test attributes
    assert restorer.__class__.__name__ == "DeOldify"
    assert isinstance(restorer.generator, DeOldifyGenerator)
    assert isinstance(restorer.discriminator, DeoldifyDiscriminator)
    assert isinstance(restorer.gan_loss, GANLoss)
    assert isinstance(restorer.perceptual_loss, PerceptualLoss)

    # prepare data
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 2, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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

    # test forward_test
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 2, 256, 256)

    # test forward_train
    outputs = restorer(**data_batch, test_mode=False)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"])
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 2, 256, 256)

    # val_step
    with torch.no_grad():
        outputs = restorer.val_step(data_batch)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 2, 256, 256)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    assert isinstance(outputs["results"], dict)
    for v in [
        "loss_gan_d_fake",
        "loss_gan_d_real",
        "loss_gan_g",
        "loss_perceptual",
        "l1_loss",
    ]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1

    assert torch.equal(
        outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
    )
    assert torch.equal(
        outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
    )
    assert torch.is_tensor(outputs["results"]["image_color_fake"])
    assert outputs["results"]["image_color_fake"].size() == (1, 2, 256, 256)

    # test disc_steps and disc_init_steps
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 2, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    test_cfg = dict()

    # build restorer
    # model_cfg["generator"]["decoder"]["x_in_c_list"]=[64, 256, 512, 1024]
    # 这个道理告诉我们，善用reverse,因为用了一次reverse导致数组翻转
    model_cfg["generator"]["decoder"]["x_in_c_list"] = [64, 256, 512, 1024]
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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
    # iter 0, 1
    for i in range(2):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        for v in ["loss_gan_d_fake", "loss_gan_d_real"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 2, 256, 256)
        # assert restorer.iteration == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        # assert restorer.iteration == i
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        log_check_list = ["loss_gan_d_fake", "loss_gan_d_real"]
        for v in log_check_list:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 2, 256, 256)
        # assert restorer.iteration == i + 1


def test_deoldify_Gonly():
    model_cfg = dict(
        type="DeOldify",
        generator=dict(
            type="DeOldifyGenerator",
            encoder=dict(
                type="ColorizationResNet",
                num_layers=101,
                pretrained=None,
                out_layers=[2, 5, 6, 7],
            ),
            mid_layers=dict(type="MidConvLayer", norm_type="NormSpectral", ni=2048),
            decoder=dict(
                type="UnetWideDecoder",
                self_attention=True,
                x_in_c_list=[64, 256, 512, 1024],
                ni=2048,
                nf_factor=2,
                norm_type="NormSpectral",
            ),
            post_layers=dict(
                type="PostLayer",
                ni=256,
                last_cross=True,
                n_classes=3,
                bottle=False,
                norm_type="NormSpectral",
                y_range=(-3.0, 3.0),
            ),
        ),
        discriminator=dict(
            type="DeoldifyDiscriminator",
            in_channels=6,
            base_channels=256,
            num_blocks=3,
            p=0.15,
        ),
        nogan_stage="Gonly",
        gan_loss=dict(
            type="GANLoss",
            gan_type="vanilla",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=0.0,
        ),
        perceptual_loss=dict(
            type="PerceptualLoss",
            layer_weights={
                "18": 20.0,
                "27": 70.0,
                "36": 10.0,
            },
            vgg_type="vgg19",
            perceptual_weight=1.0,
            style_weight=0,
            criterion="mse",
        ),
        l1_loss=dict(loss_weight=1.0),
    )
    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    # test attributes
    assert restorer.__class__.__name__ == "DeOldify"
    assert isinstance(restorer.generator, DeOldifyGenerator)
    assert isinstance(restorer.discriminator, DeoldifyDiscriminator)
    assert isinstance(restorer.gan_loss, GANLoss)
    assert isinstance(restorer.perceptual_loss, PerceptualLoss)

    # prepare data
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 3, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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

    # test forward_test
    with torch.no_grad():
        outputs = restorer(**data_batch, test_mode=True)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # test forward_train
    outputs = restorer(**data_batch, test_mode=False)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"])
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # val_step
    with torch.no_grad():
        outputs = restorer.val_step(data_batch)
    assert torch.equal(outputs["img_color_real"], data_batch["img_color"].cpu())
    assert torch.is_tensor(outputs["img_color_fake"])
    assert outputs["img_color_fake"].size() == (1, 3, 256, 256)

    # test train_step
    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    assert isinstance(outputs["results"], dict)
    for v in ["loss_perceptual", "l1_loss"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1

    assert torch.equal(
        outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
    )
    assert torch.equal(
        outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
    )
    assert torch.is_tensor(outputs["results"]["image_color_fake"])
    assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)

    # test disc_steps and disc_init_steps
    img_gray = torch.rand(1, 3, 256, 256).cuda()
    img_color = torch.rand(1, 3, 256, 256).cuda()
    data_batch = {"img_gray": img_gray, "img_color": img_color, "meta": dict()}

    train_cfg = dict(disc_steps=2, disc_init_steps=2)
    test_cfg = dict()

    # build restorer
    # model_cfg["generator"]["decoder"]["x_in_c_list"]=[64, 256, 512, 1024]
    # 这个道理告诉我们，善用reverse,因为用了一次reverse导致数组翻转
    model_cfg["generator"]["decoder"]["x_in_c_list"] = [64, 256, 512, 1024]
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.5, 0.999))
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
    # iter 0, 1
    for i in range(2):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)
        # assert restorer.iteration == i + 1

    # iter 2, 3, 4, 5
    for i in range(2, 6):
        # assert restorer.iteration == i
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["results"], dict)
        assert outputs["num_samples"] == 1
        assert torch.equal(
            outputs["results"]["image_color_real"], data_batch["img_color"].cpu()
        )
        assert torch.equal(
            outputs["results"]["image_gray_real"], data_batch["img_gray"].cpu()
        )
        assert torch.is_tensor(outputs["results"]["image_color_fake"])
        assert outputs["results"]["image_color_fake"].size() == (1, 3, 256, 256)
