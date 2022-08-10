from __future__ import absolute_import, division, print_function

import argparse
import os
import random
from glob import glob

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mmdp.models.architectures.generator.dvp_arch import UNet as UnetGenerator
from mmdp.models.architectures.generator.vgg_arch import VGGFeatureExtractor as vgg19

# from mmdp.models.architectures.generator.Unet_net import UnetGenerator

# from torchstat import stat


def parse_args():
    parser = argparse.ArgumentParser(description="deep-video-prior")
    parser.add_argument("--task", default="colorization", type=str, help="Name of task")
    parser.add_argument("--use_gpu", default=1, type=int, help="use gpu or not")
    parser.add_argument("--max", default=20, type=int, help="epoch+1")
    parser.add_argument(
        "--input",
        default="../catandmouse_c",
        type=str,
        help="dir of input images",
    )
    parser.add_argument(
        "--processed",
        default="../catandmouse_b",
        type=str,
        help="dir of processed images",
    )
    parser.add_argument(
        "--output", default="None", type=str, help="dir of output video"
    )
    parser.add_argument(
        "--res_name",
        default="test",
        type=str,
        help="name of output video and output image dir",
    )
    args = parser.parse_args()
    return args


def compute_error(real, fake):
    # return tf.reduce_mean(tf.abs(fake-real))
    return torch.mean(torch.abs(fake - real))


def prepare_paired_input(task, id, input_names, processed_names, is_train=0):
    net_in = np.float32(mmcv.imread(input_names[id])) / 255.0
    if len(net_in.shape) == 2:
        net_in = np.tile(net_in[:, :, np.newaxis], [1, 1, 3])
    net_gt = np.float32(mmcv.imread(processed_names[id])) / 255.0
    org_h, org_w = net_in.shape[:2]
    h = org_h // 32 * 32
    w = org_w // 32 * 32
    #     net_gt = cv2.resize(net_gt,(w,h))
    # print(net_in.shape, net_gt.shape)
    return net_in[np.newaxis, :h, :w, :], net_gt[np.newaxis, :h, :w, :]


# def read_img():
#     args = parse_args()
#     img_path = args.input
#     img = cv2.imread(img_path)
#     h,w,_ = img.shape
#     size = (w,h)
#     return size
# some functions


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # nn.init.kaiming_normal_(module.weight)
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()


def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std


def main():
    seed = 2020
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # process arguments
    args = parse_args()
    max = args.max
    input_folder = args.input
    processed_folder = args.processed
    task = args.task
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    #     size = read_img()
    L1 = torch.nn.L1Loss()
    # define loss function
    layer_name_list = {"relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"}
    VGG_19 = vgg19(layer_name_list, use_input_norm=False).to(device)

    def Lp_loss(x, y):
        vgg_real = VGG_19(normalize_batch(x))
        vgg_fake = VGG_19(normalize_batch(y))
        p0 = compute_error(normalize_batch(x), normalize_batch(y))
        content_loss_list = []
        content_loss_list.append(p0)
        feat_layers = {
            "relu1_2": 1.0 / 2.6,
            "relu2_2": 1.0 / 4.8,
            "relu3_2": 1.0 / 3.7,
            "relu4_2": 1.0 / 5.6,
            "relu5_2": 10.0 / 1.5,
        }

        for layer, w in feat_layers.items():
            pi = compute_error(vgg_real[layer], vgg_fake[layer])
            content_loss_list.append(w * pi)

        content_loss = torch.sum(torch.stack(content_loss_list))
        return content_loss

    # Define model .
    out_channels = 3
    # net = UnetGenerator(
    #     in_channels=3, out_channels=out_channels, base_channels=32, kernel3=True
    # )
    net = UnetGenerator(in_channels=3, out_channels=out_channels, init_features=32)
    net.to(device)
    # stat(net, (3,512,512))
    # print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000,8000], gamma=0.5)
    # prepare data
    input_folders = [input_folder]
    processed_folders = [processed_folder]
    # start to train
    for folder_idx, input_folder in enumerate(input_folders):
        # -----------load data-------------
        input_names = sorted(glob(input_folders[folder_idx] + "/*"))
        processed_names = sorted(glob(processed_folders[folder_idx] + "/*"))
        if args.output == "None":
            output_folder = "./result/{}".format(task + "/" + args.res_name)
        else:
            output_folder = args.output + "/" + task + "/" + args.res_name
        # print(output_folder, input_folders[folder_idx], processed_folders[folder_idx])
        num_of_sample = min(len(input_names), len(processed_names))
        initialize_weights(net)
        #         step = 0
        for epoch in range(0, max):
            frame_id = 0
            if not os.path.isdir("{}".format(output_folder)):
                os.makedirs("{}".format(output_folder))
            # -----------start to train-------------
            print("Processing epoch {}".format(epoch))
            # print(len(input_names), len(processed_names))
            for id in tqdm(range(num_of_sample)):
                net_in, net_gt = prepare_paired_input(
                    task, id, input_names, processed_names
                )
                net_in_tensor = (
                    torch.from_numpy(net_in).permute(0, 3, 1, 2).float().to(device)
                )
                n, c, h, w = net_in_tensor.shape
                size = (w, h)
                net_gt_tensor = (
                    torch.from_numpy(net_gt).permute(0, 3, 1, 2).float().to(device)
                )
                prediction = net(net_in_tensor)
                crt_loss = 0.5 * Lp_loss(prediction, net_gt_tensor) + 0.5 * L1(
                    prediction, net_gt_tensor
                )
                optimizer.zero_grad()
                crt_loss.backward()
                optimizer.step()
                frame_id += 1

            # # -----------save intermidiate results-------------
            if epoch == (max - 1):
                print("\n------ 生成视频中 ------")
                video = cv2.VideoWriter(
                    output_folder + "/" + args.res_name + ".mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25,
                    size,
                )
                for id in tqdm(range(num_of_sample)):
                    net_in, net_gt = prepare_paired_input(
                        task, id, input_names, processed_names
                    )
                    net_in_tensor = (
                        torch.from_numpy(net_in).permute(0, 3, 1, 2).float().to(device)
                    )  # Option:

                    with torch.no_grad():
                        prediction = net(net_in_tensor)
                    prediction = prediction.detach().permute(0, 2, 3, 1).cpu().numpy()
                    h, w, _ = prediction[0].shape
                    img = np.uint8(prediction[0].clip(0, 1) * 255.0)
                    #                     img = cv2.resize(img, size)
                    video.write(img)
                video.release()


if __name__ == "__main__":
    main()
    print("视频生成完毕，请查看results下的mp4文件")
