import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# added
from mmcv import Config

import mmdp.models.util_pix as util
from mmdp.models import build_model

if __name__ == "__main__":
    opt = Config.fromfile("./configs/mmcolor/color_siggraph_regre2.py")
    # build model
    model = build_model(opt.model, train_cfg=opt.train_cfg, test_cfg=opt.test_cfg)
    path = "./work_dirs/color_siggraph_regre2/ckpt/color_siggraph_regre2/latest.pth"
    state_dict = torch.load(path)
    model.load_state_dict(state_dict["state_dict"], strict=True)
    opt.batch_size = 1  # test code only supports batch_size = 1
    # opt.dataroot = './dataset/ilsvrc2012/train/'
    opt.dataroot = "./data/ILSVRC/train_small"
    opt.how_many = 20
    # build datasets
    dataset = torchvision.datasets.ImageFolder(
        opt.dataroot,
        transform=transforms.Compose(
            [transforms.Resize((opt.load_size, opt.load_size)), transforms.ToTensor()]
        ),
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )
    # statistics
    psnrs = np.zeros((opt.how_many, 1))
    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)
        # with no points
        img_path = [str.replace("%08d" % (i), ".", "p")]
        result = model.forward(data_raw[0], psnr=True)
        visuals = model.get_current_visuals()
        psnrs[i] = util.calculate_psnr_np(
            util.tensor2im(visuals["real"]), util.tensor2im(visuals["fake_reg"])
        )
        if i % 5 == 0:
            print("processing (%04d)-th image... %s" % (i, img_path))

        if i == opt.how_many - 1:
            break

    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    print("%.2f+/-%.2f" % (psnrs_mean, psnrs_std))
