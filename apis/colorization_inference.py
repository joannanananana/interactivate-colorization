# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from mmcv.parallel import collate, scatter
from PIL import Image

from mmdp.datasets.pipelines import Compose
from mmdp.models import build_model


def init_colorization_model(config, checkpoint=None, device="cuda:0"):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.
    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError(
            "config must be a filename or Config object, " f"but got {type(config)}"
        )
    # config.model.pretrained = None
    # config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    params = torch.load(checkpoint, map_location="cpu")

    if "state_dict" in params:
        model.load_state_dict(params["state_dict"])
    else:
        generator = model.generator
        # checkpoint = load_checkpoint(model, checkpoint)

        # ------------generator keys transform------------------
        keys_0 = generator.state_dict().keys()
        # torch.save({'generator': generator.state_dict()}, 'keys.pth')

        keys_1 = params["model"].keys()
        # print(keys_0 == keys_1)

        if keys_0 != keys_1 and len(keys_0) == len(keys_1):
            d = params["model"].items()
            d1 = generator.state_dict().items()
            from collections import OrderedDict

            new_d = OrderedDict()
            for (k, v), (k1, v1) in zip(d, d1):
                new_d[k1] = v
            params["model"] = new_d
        # ------------------------------

        generator.load_state_dict(params["model"])

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def denorm(x, mean, std, deoldify_pretrained=False):
    if deoldify_pretrained:
        x = x * std[..., None, None] + mean[..., None, None]
    else:
        x = (x + 1) / 2
        x = x[[2, 1, 0], :, :]
    return x


def overlay_color(generated_color_iamge, gray_image):
    color_y, color_u, color_v = generated_color_iamge.convert("YCbCr").split()
    orig_y, orig_u, orig_v = gray_image.convert("YCbCr").split()
    final = Image.merge("YCbCr", (orig_y, color_u, color_v)).convert("RGB")
    return final


def post_process(result, img_gray, deoldify_pretrained=False):
    # 这里只能处理一张图，可能要改
    results = result["img_color_fake"].squeeze(0)

    # denom
    mean = torch.tensor([0.4850, 0.4560, 0.4060])  # imagenet的均值和方差
    std = torch.tensor([0.2290, 0.2240, 0.2250])
    results = denorm(results.detach(), mean, std, deoldify_pretrained)

    # clamp
    results = results.float().clamp(min=0, max=1)

    # To PIL
    out = (results.cpu().numpy() * 255).astype("uint8").transpose(1, 2, 0)
    out = Image.fromarray(out)

    # img_gray = (img_gray.cpu().numpy()*255).astype('uint8').transpose(1, 2, 0)
    # img_gray = Image.fromarray(img_gray)

    # Resize
    orig_image = None
    if isinstance(img_gray, str):
        orig_image = PIL.Image.open(img_gray).convert("RGB")
    if isinstance(img_gray, np.ndarray):
        orig_image = Image.fromarray(np.uint8(img_gray))

    raw_color = out.resize(orig_image.size, resample=PIL.Image.BILINEAR)

    # overlay color
    final = overlay_color(raw_color, orig_image)

    # # return final
    # if isinstance(img_gray, str):
    #     return final
    # if isinstance(img_gray, np.ndarray):
    #     return np.asarray(final)[:, :, ::-1]
    # return None
    if isinstance(img_gray, str):
        return transforms.ToTensor()(final)
    if isinstance(img_gray, np.ndarray):
        return np.asarray(final)[:, :, ::-1]
    return None


def colorization_inference(model, img, deoldify_pretrained=False):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.
    Returns:
        np.ndarray: The predicted colorization result.
    """

    # cfg = model.cfg
    device = next(model.parameters()).device  # model device

    # build the data pipeline
    if model.cfg.get("demo_pipeline", None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get("test_pipeline", None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    # remove gt from test_pipeline
    # 待补充，下面这一段不一定对
    keys_to_remove = ["gt", "gt_path"]
    for key in keys_to_remove:
        for pipeline in list(test_pipeline):
            if "key" in pipeline and key == pipeline["key"]:
                test_pipeline.remove(pipeline)
            if "keys" in pipeline and key in pipeline["keys"]:
                pipeline["keys"].remove(key)
                if len(pipeline["keys"]) == 0:
                    test_pipeline.remove(pipeline)
            if "meta_keys" in pipeline and key in pipeline["meta_keys"]:
                pipeline["meta_keys"].remove(key)

    # build the data pipeline
    test_pipeline = Compose(test_pipeline)

    # prepare data
    data = None
    if isinstance(img, str):
        data = dict(img_gray_path=img)
    if isinstance(img, np.ndarray):
        data = dict(img_gray=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # # forward the model
    # model.eval()
    # with torch.no_grad():
    #     results = model.forward(data['gray_img']).squeeze()

    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    final = post_process(result, img, deoldify_pretrained)

    return final
