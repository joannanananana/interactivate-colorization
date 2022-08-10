import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

from mmdp.utils import colorize_image as CI
from mmdp.utils.img_util import imwrite

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Inference demo for QuanTexSR."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./data/ILSVRC/test/dog.jpeg",
        help="Input image or folder",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=str,
        default="./work_dirs/color_siggraph_train/ckpt/color_siggraph_train/latest.pth",
        help="path for model weights",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./results/",
        help="Output folder",
    )
    args = parser.parse_args()
    # Choose gpu to run the model on
    gpu_id = 0
    # Initialize colorization class
    colorModel = CI.ColorizeImageTorch(Xd=256, maskcent="pytorch_maskcent")
    # Load the model
    colorModel.prep_net(gpu_id, args.weight)
    # '../work_dirs/color_siggraph_train_small/ckpt/color_siggraph_train_small/iter_1700.pth'
    # Load the image
    # '../data/ILSVRC/test/dog.jpeg'

    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, "*")))

    pbar = tqdm(total=len(paths), unit="image")
    for _, path in enumerate(paths):
        img_name = os.path.basename(path)
        colorModel.load_image(args.input)  # load an image
        mask = np.zeros((1, 256, 256))  # giving no user points, so mask is all 0's
        input_ab = np.zeros(
            (2, 256, 256)
        )  # ab values of user points, default to 0 for no input
        img_out = colorModel.net_forward(
            input_ab, mask, color_mode=True
        )  # run model, returns 256x256 image
        save_path = os.path.join(args.output, f"{img_name}")
        imwrite(img_out, save_path)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
