_base_ = [
    "../_base_/datasets/paired_imgs_256x256_crop.py",
    "../_base_/default_runtime.py",
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
domain_a = "gray"
domain_b = "color"
model = dict(
    type="Pix2Pix",
    generator=dict(
        type="SIGGRAPHGenerator", input_nc=4, output_nc=2, classification=False
    ),
    gan_loss=dict(type="GANLoss", gan_type="lsgan"),
    l1_loss=dict(type="L1Loss"),
    l2_loss=dict(type="L2Loss"),
    lambda_A=1,
)
ab_quant = 10.0  # quantization factor
ab_max = 110.0  # maximimum ab value
ab_norm = 110.0  # colorization normalization factorpip
A = 2 * ab_max / ab_quant + 1
B = A

# model training and testing settings
# + mask_cent信息
train_cfg = dict(ab_quant=10.0, ab_max=110.0, ab_norm=110.0, A=A, B=B, sample_p=1.0)
test_cfg = dict(
    A=1.0,
    l_norm=100.0,
    l_cent=50,
    ab_norm=110,
)

lr_config = dict(
    policy="step", by_epoch=False, step=[50000, 100000, 200000, 300000], gamma=0.5
)

checkpoint_config = dict(interval=100, save_optimizer=True, by_epoch=False)

log_config = dict()

load_size = 256

dataroot = "/openbayes/input/input2/ILSVRC"

val_dataset_type = "PairedImageDataset"
loadsize = 256
finesize = 176
train_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="pair"),
    dict(
        type="Resize",
        keys=["pair"],
        backend="cv2",
        scale=(loadsize, loadsize),
        interpolation="bilinear",
    ),
    dict(type="Crop", keys=["pair"], crop_size=(finesize, finesize), random_crop=True),
    dict(type="Flip", keys=["pair"], flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg, keys=["pair"]),
    dict(type="ImageToTensor", keys=["pair"]),
    dict(type="Collect", keys=["pair"], meta_keys=["pair_path"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile", io_backend="disk", key="pair"),
    dict(
        type="Resize",
        keys=["pair"],
        backend="cv2",
        scale=(loadsize, loadsize),
        interpolation="bilinear",
    ),
    dict(
        type="ToTensor",
        keys=["pair"],
    ),
    dict(type="Collect", keys=["pair"], meta_keys=["pair_path"]),
]
batch_size = 32
total_iters = 142677
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    shuffle=True,
    train=dict(dataroot=dataroot, pipeline=train_pipeline),
    val=dict(dataroot=dataroot, pipeline=test_pipeline),
    test=dict(dataroot=dataroot, pipeline=test_pipeline, test_mode=True),
)

data_test = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    shuffle=True,
    train=dict(dataroot=dataroot, pipeline=train_pipeline),
    val=dict(dataroot=dataroot, pipeline=test_pipeline),
    test=dict(dataroot=dataroot, pipeline=test_pipeline),
)
optimizer = dict(
    generator=dict(type="Adam", lr=1e-5, betas=(0.9, 0.999)),
)

runner = None
use_ddp_wrapper = True
l_norm = 100.0
l_cent = 50.0
ab_norm = 110.0
load_from = "./work_dirs/color_siggraph_train/ckpt/color_siggraph_train/latest.pth"
