source_domain = None  # set by user
target_domain = None  # set by user
# model settings
model = dict(
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
        init_cfg=dict(type="normal", gain=0.02),
    ),
    gan_loss=dict(
        type="GANLoss",
        gan_type="lsgan",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0,
    ),
    perceptual_loss=dict(
        type="PerceptualLoss",
        layer_weights={"29": 1.0},
        vgg_type="vgg19",
        perceptual_weight=1e-2,
        style_weight=0,
        criterion="mse",
    ),
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict()
