import argparse
import os
import shutil
import sys

import mmcv
from mmcv import DictAction
from torchvision import utils

# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmdp.apis import init_model, sample_img2img_model  # isort:skip  # noqa
# yapf: enable


def parse_args():
    parser = argparse.ArgumentParser(description="Translation demo")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("video_path", help="Video file path")
    parser.add_argument("start", help="Video start frame")
    parser.add_argument("end", help="Video end frame")
    parser.add_argument(
        "--target-domain", type=str, default=None, help="Desired image domain"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./work_dirs/demos/translation_sample.png",
        help="path to save translation sample",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device id")
    # args for inference/sampling
    parser.add_argument(
        "--sample-cfg",
        nargs="+",
        action=DictAction,
        help="Other customized kwargs for sampling function",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = init_model(args.config, checkpoint=args.checkpoint, device=args.device)

    if args.sample_cfg is None:
        args.sample_cfg = dict()

    mmcv.cut_video(
        args.video_path,
        "./tmp.mp4",
        start=int(args.start),
        end=int(args.end),
        vcodec="h264",
    )
    video = mmcv.VideoReader("./tmp.mp4")
    # video = video[int(args.start):int(args.end)]

    tmp_dir = "./tmp/"
    mmcv.mkdir_or_exist(os.path.dirname(tmp_dir))
    video.cvt2frames(tmp_dir)

    ret_dir = "./ret/"
    mmcv.mkdir_or_exist(os.path.dirname(ret_dir))
    shutil.rmtree(ret_dir)
    mmcv.mkdir_or_exist(os.path.dirname(ret_dir))
    idx = 0

    for img in video:
        ret = sample_img2img_model(
            model,
            os.path.join(tmp_dir, f"{idx:06}.jpg"),
            args.target_domain,
            None,
            **args.sample_cfg,
        )
        ret = (ret[:, [2, 1, 0]] + 1.0) / 2.0

        # save images
        utils.save_image(ret, os.path.join(ret_dir, f"{idx:06}.jpg"))
        idx = idx + 1

    mmcv.frames2video(ret_dir, "result_video.avi")


if __name__ == "__main__":
    main()
