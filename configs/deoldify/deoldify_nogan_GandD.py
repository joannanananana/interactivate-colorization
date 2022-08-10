# custom_imports = dict(
#   imports=['models', 'apis'],
#   allow_failed_imports=False)
_base_ = [
    "../_base_/datasets/paired_imgs_256x256_crop.py",
    "../_base_/default_runtime.py",
]

domain_a = "gray"
domain_b = "color"

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

train_cfg = dict()
test_cfg = dict()

dataroot = "./data/ILSVRC"
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

train_pipeline = [
    dict(
        type="LoadPairedImageFromFile",
        io_backend="disk",
        key="pair",
        domain_a=domain_a,
        domain_b=domain_b,
        flag="color",
    ),
    dict(
        type="Resize",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        scale=(256, 256),
        interpolation="bicubic",
    ),
    dict(
        type="FixedCrop",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        crop_size=(192, 192),
    ),
    dict(
        type="Flip", keys=[f"img_{domain_a}", f"img_{domain_b}"], direction="horizontal"
    ),
    dict(type="RescaleToZeroOne", keys=[f"img_{domain_a}", f"img_{domain_b}"]),
    dict(
        type="Normalize",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    dict(type="ImageToTensor", keys=[f"img_{domain_a}", f"img_{domain_b}"]),
    dict(
        type="Collect",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        meta_keys=[f"img_{domain_a}_path", f"img_{domain_b}_path"],
    ),
]

test_pipeline = [
    dict(
        type="LoadPairedImageFromFile",
        io_backend="disk",
        key="pair",
        domain_a=domain_a,
        domain_b=domain_b,
        flag="color",
    ),
    dict(
        type="Resize",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        scale=(192, 192),
        interpolation="bicubic",
    ),
    dict(type="RescaleToZeroOne", keys=[f"img_{domain_a}", f"img_{domain_b}"]),
    dict(
        type="Normalize",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    dict(type="ImageToTensor", keys=[f"img_{domain_a}", f"img_{domain_b}"]),
    dict(
        type="Collect",
        keys=[f"img_{domain_a}", f"img_{domain_b}"],
        meta_keys=[f"img_{domain_a}_path", f"img_{domain_b}_path"],
    ),
]

demo_pipeline = [
    dict(type="Resize", keys=["img_gray"], scale=(192, 192), interpolation="bicubic"),
    dict(type="RescaleToZeroOne", keys=["img_gray"]),
    dict(
        type="Normalize",
        keys=["img_gray"],
        to_rgb=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    # mean=[0.4850, 0.4560, 0.4060],
    # std=[0.2290, 0.2240, 0.2250],
    dict(type="ImageToTensor", keys=["img_gray"]),
    dict(type="Collect", keys=["img_gray"], meta_keys=["img_gray"]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    shuffle=True,
    train=dict(dataroot=dataroot, pipeline=train_pipeline),
    val=dict(dataroot=dataroot, pipeline=test_pipeline),  # testdir='val'),
    test=dict(dataroot=dataroot, pipeline=test_pipeline),
)  # , testdir='val'))

optimizer = dict(
    # generator=dict(type='Adam', lr=1e-4, betas=(0.5, 0.999)),
    # discriminator=dict(type='Adam', lr=1e-4, betas=(0.5, 0.999)))
    generator=dict(type="Adam", lr=5e-6, betas=(0.0, 0.999), weight_decay=1e-3),
    discriminator=dict(type="Adam", lr=2.5e-5, betas=(0.0, 0.999), weight_decay=1e-3),
)

lr_config = None

## comment out during pretrain phase
# learning policy
# lr_config = dict(
#   policy='OneCycle',
#   max_lr=4e-4,
#   #pct_start=1e-8,
#   pct_start=0.3,
#   anneal_strategy='cos',
#   div_factor=25,
#   final_div_factor=100000.0,
#   three_phase=False)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
custom_hooks = [
    dict(
        type="MMGenVisualizationHook",
        output_dir="training_samples",
        test_output_dir="testing_samples",
        res_name_list=[f"fake_{domain_a}", f"fake_{domain_b}"],
        interval=200,
    )
]

log_config = dict(interval=100)

runner = None
# use dynamic runner
#   runner = dict(
#       type='DynamicIterBasedRunner',
#       is_dynamic_ddp=True,
#       pass_training_status=True)

use_ddp_wrapper = True

total_iters = 610000
# workflow = [('train', 1)]
workflow = [("train", 5000), ("val", 2)]
exp_name = "deoldify_pytorch_gray2color"
work_dir = f"./work_dirs/experiments/{exp_name}"
