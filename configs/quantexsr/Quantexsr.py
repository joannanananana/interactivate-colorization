scale = 4
# model settings
model = dict(
    type="QuanTexSRGANModel",
    generator=dict(
        type="FeMaSRNet",
        in_channel=3,
        beta=0.25,
        scale_factor=scale,
        gt_resolution=256,
        norm_type="gn",
        act_type="silu",
        frozen_module_keywords=["quantize"],
        LQ_stage=True,
    ),
    discriminator=dict(type="UNetDiscriminatorSN", num_in_ch=3, skip_connection=True),
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
)

# model training and testing settings
train_cfg = dict()
test_cfg = dict(metrics=["PSNR", "SSIM", "Lpips"], crop_border=scale)

# dataset settings
train_dataset_type = "SRFolderDataset"
val_dataset_type = "SRFolderDataset"
train_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="lq", flag="unchanged"),
    dict(type="LoadImageFromFile", io_backend="disk", key="gt", flag="unchanged"),
    # dict(type="Lq_degradation_bsrgan", key="lq", sf=4),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(
        type="Normalize", keys=["lq", "gt"], mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True
    ),
    dict(type="PairedRandomCrop", gt_patch_size=128),
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="horizontal"),
    dict(type="Flip", keys=["lq", "gt"], flip_ratio=0.5, direction="vertical"),
    dict(type="RandomTransposeHW", keys=["lq", "gt"], transpose_ratio=0.5),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "gt_path"]),
    dict(type="ImageToTensor", keys=["lq", "gt"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="lq", flag="unchanged"),
    dict(type="LoadImageFromFile", io_backend="disk", key="gt", flag="unchanged"),
    # dict(type="Lq_util", key="lq", sf=4),
    dict(type="RescaleToZeroOne", keys=["lq", "gt"]),
    dict(
        type="Normalize", keys=["lq", "gt"], mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True
    ),
    dict(type="Collect", keys=["lq", "gt"], meta_keys=["lq_path", "lq_path"]),
    dict(type="ImageToTensor", keys=["lq", "gt"]),
]

data = dict(
    workers_per_gpu=8,
    samples_per_gpu=4,
    shuffle=True,
    train=dict(
        type="RepeatDataset",
        times=1000,
        pipeline=train_pipeline,
        dataset=dict(
            type=train_dataset_type,
            lq_folder="./data/Quantexsr_demo/lq",
            gt_folder="./data/Quantexsr_demo/gt",
            pipeline=train_pipeline,
            scale=scale,
        ),
    ),
    val=dict(
        type=train_dataset_type,
        lq_folder="./data/Quantexsr_demo/lq",
        gt_folder="./data/Quantexsr_demo/gt",
        pipeline=train_pipeline,
        scale=scale,
        filename_tmpl="{}",
    ),
    test=dict(
        type=val_dataset_type,
        lq_folder="./data/Quantexsr_demo/lq",
        gt_folder="./data/Quantexsr_demo/gt",
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl="{}",
    ),
)

# optimizer
optimizer = dict(
    generator=dict(type="Adam", lr=1e-4, betas=(0.9, 0.999)),
    discriminator=dict(type="Adam", lr=1e-4, betas=(0.9, 0.999)),
)

# learning policy
total_iters = 1000
lr_config = dict(
    policy="Step", by_epoch=False, step=[500, 1000, 20000, 30000], gamma=0.5
)

checkpoint_config = dict(interval=1000, save_optimizer=True, by_epoch=False)

log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
        # dict(type='PaviLoggerHook', init_kwargs=dict(project='mmedit-sr'))
    ],
)
visual_config = None

evaluation = dict(
    type="EvalIterHook",
    interval=100,
    save_image=True,
    save_path="./work_dirs/Quantexsr/img/",
)

# runtime settings
log_level = "INFO"  # 日志级别
work_dir = "./work_dirs/Quantexsr/"  # 保存当前实验的模型检查点和日志的目录
# load_from = "/openbayes/input/input2/pretrain_model_latest.pth"
load_from = "/openbayes/input/input1/FeMaSR_SRX4_model_g.pth"
resume_from = None
workflow = [("train", 2), ("val", 1)]  # train 2 times,val 1 times
