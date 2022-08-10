import os
from os import path as osp

from PIL import Image


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def generate_meta_info():
    """Generate meta info for dataset."""

    gt_folder = "/home/iecy/pycharm/Openbayes/obvisionflow_2/data/Quantexsr/train/HQ_sub_samename/"
    meta_info_txt = "/home/iecy/pycharm/Openbayes/obvisionflow_2/data/Quantexsr/train/HQ_sub_samename/meta_info_GT.txt"

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, "w") as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == "RGB":
                n_channel = 3
            elif mode == "L":
                n_channel = 1
            else:
                raise ValueError(f"Unsupported mode {mode}.")

            info = f"{img_path} ({height},{width},{n_channel})"
            print(idx + 1, info)
            f.write(f"{info}\n")


if __name__ == "__main__":
    generate_meta_info()
