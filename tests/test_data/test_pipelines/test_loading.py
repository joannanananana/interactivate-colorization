# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from pathlib import Path

import mmcv
import numpy as np
import pytest

from mmdp.datasets.pipelines import LoadImageFromFile, LoadPairedImageFromFile


def test_load_image_from_file():
    path_baboon = Path(__file__).parent.parent.parent / "data" / "gt" / "baboon.png"
    img_baboon = mmcv.imread(str(path_baboon), flag="color")
    path_baboon_x4 = (
        Path(__file__).parent.parent.parent / "data" / "lq" / "baboon_x4.png"
    )
    img_baboon_x4 = mmcv.imread(str(path_baboon_x4), flag="color")

    # read gt image
    # input path is Path object
    results = dict(gt_path=path_baboon)
    config = dict(io_backend="disk", key="gt")
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results["gt"].shape == (480, 500, 3)
    np.testing.assert_almost_equal(results["gt"], img_baboon)
    assert results["gt_path"] == str(path_baboon)
    # input path is str
    results = dict(gt_path=str(path_baboon))
    results = image_loader(results)
    assert results["gt"].shape == (480, 500, 3)
    np.testing.assert_almost_equal(results["gt"], img_baboon)
    assert results["gt_path"] == str(path_baboon)

    # read lq image
    # input path is Path object
    results = dict(lq_path=path_baboon_x4)
    config = dict(io_backend="disk", key="lq")
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results["lq"].shape == (120, 125, 3)
    np.testing.assert_almost_equal(results["lq"], img_baboon_x4)
    assert results["lq_path"] == str(path_baboon_x4)
    # input path is str
    results = dict(lq_path=str(path_baboon_x4))
    results = image_loader(results)
    assert results["lq"].shape == (120, 125, 3)
    np.testing.assert_almost_equal(results["lq"], img_baboon_x4)
    assert results["lq_path"] == str(path_baboon_x4)
    assert repr(image_loader) == (
        image_loader.__class__.__name__
        + (
            "(io_backend=disk, key=lq, "
            "flag=color, save_original_img=False, channel_order=bgr, "
            "use_cache=False)"
        )
    )

    results = dict(lq_path=path_baboon_x4)
    config = dict(io_backend="disk", key="lq", flag="grayscale", save_original_img=True)
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results["lq"].shape == (120, 125)
    assert results["lq_ori_shape"] == (120, 125)
    np.testing.assert_almost_equal(results["ori_lq"], results["lq"])
    assert id(results["ori_lq"]) != id(results["lq"])

    # test: use_cache
    results = dict(gt_path=path_baboon)
    config = dict(io_backend="disk", key="gt", use_cache=True)
    image_loader = LoadImageFromFile(**config)
    assert image_loader.cache is None
    assert repr(image_loader) == (
        image_loader.__class__.__name__
        + (
            "(io_backend=disk, key=gt, "
            "flag=color, save_original_img=False, channel_order=bgr, "
            "use_cache=True)"
        )
    )
    results = image_loader(results)
    assert image_loader.cache is not None
    assert str(path_baboon) in image_loader.cache
    assert results["gt"].shape == (480, 500, 3)
    assert results["gt_path"] == str(path_baboon)
    np.testing.assert_almost_equal(results["gt"], img_baboon)

    # convert to y-channel (bgr2y)
    results = dict(gt_path=path_baboon)
    config = dict(io_backend="disk", key="gt", convert_to="y")
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results["gt"].shape == (480, 500, 1)
    img_baboon_y = mmcv.bgr2ycbcr(img_baboon, y_only=True)
    img_baboon_y = np.expand_dims(img_baboon_y, axis=2)
    np.testing.assert_almost_equal(results["gt"], img_baboon_y)
    assert results["gt_path"] == str(path_baboon)

    # convert to y-channel (rgb2y)
    results = dict(gt_path=path_baboon)
    config = dict(io_backend="disk", key="gt", channel_order="rgb", convert_to="y")
    image_loader = LoadImageFromFile(**config)
    results = image_loader(results)
    assert results["gt"].shape == (480, 500, 1)
    img_baboon_y = mmcv.bgr2ycbcr(img_baboon, y_only=True)
    img_baboon_y = np.expand_dims(img_baboon_y, axis=2)
    np.testing.assert_almost_equal(results["gt"], img_baboon_y)
    assert results["gt_path"] == str(path_baboon)

    # convert to y-channel (ValueError)
    results = dict(gt_path=path_baboon)
    config = dict(io_backend="disk", key="gt", convert_to="abc")
    image_loader = LoadImageFromFile(**config)
    with pytest.raises(ValueError):
        results = image_loader(results)


class TestGenerationLoading:
    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.pair_path = osp.join("./tests", "data/paired/train/1.jpg")
        cls.results = dict(pair_path=cls.pair_path)
        cls.pair_img = mmcv.imread(str(cls.pair_path), flag="color")
        w = cls.pair_img.shape[1]
        new_w = w // 2
        cls.img_a = cls.pair_img[:, :new_w, :]
        cls.img_b = cls.pair_img[:, new_w:, :]
        cls.pair_shape = cls.pair_img.shape
        cls.img_shape = cls.img_a.shape
        cls.pair_shape_gray = (256, 512, 3)
        cls.img_shape_gray = (256, 256, 3)

    def test_load_paired_image_from_file(self):
        # RGB
        target_keys = [
            "pair_path",
            "pair",
            "pair_ori_shape",
            "img_gray",
            "img_color",
            "img_gray_path",
            "img_color_path",
            "img_gray_ori_shape",
            "img_color_ori_shape",
        ]
        domain_a = "gray"
        domain_b = "color"
        config = dict(
            io_backend="disk",
            key="pair",
            domain_a=domain_a,
            domain_b=domain_b,
            flag="color",
        )
        results = copy.deepcopy(self.results)
        load_paired_image_from_file = LoadPairedImageFromFile(**config)
        results = load_paired_image_from_file(results)

        assert self.check_keys_contain(results.keys(), target_keys)
        assert results["pair"].shape == self.pair_shape
        assert results["pair_ori_shape"] == self.pair_shape
        np.testing.assert_equal(results["pair"], self.pair_img)
        assert results["pair_path"] == self.pair_path
        assert results["img_color"].shape == self.img_shape
        assert results["img_color_ori_shape"] == self.img_shape
        np.testing.assert_equal(results["img_color"], self.img_b)
        assert results["img_color_path"] == self.pair_path
        assert results["img_gray"].shape == self.img_shape
        assert results["img_gray_ori_shape"] == self.img_shape
        np.testing.assert_equal(results["img_gray"], self.img_a)
        assert results["img_gray_path"] == self.pair_path

        # Grayscale & save_original_img
        target_keys = [
            "pair_path",
            "pair",
            "pair_ori_shape",
            "ori_pair",
            "img_color_path",
            "img_color",
            "img_color_ori_shape",
            "ori_img_color",
            "img_gray_path",
            "img_gray",
            "img_gray_ori_shape",
            "ori_img_gray",
        ]
        config = dict(
            io_backend="disk",
            key="pair",
            domain_a=domain_a,
            domain_b=domain_b,
            flag="color",
            save_original_img=True,
        )
        results = copy.deepcopy(self.results)
        load_paired_image_from_file = LoadPairedImageFromFile(**config)
        results = load_paired_image_from_file(results)

        assert self.check_keys_contain(results.keys(), target_keys)
        assert results["pair"].shape == self.pair_shape_gray
        assert results["pair_ori_shape"] == self.pair_shape_gray
        np.testing.assert_equal(results["pair"], results["ori_pair"])
        assert id(results["ori_pair"]) != id(results["pair"])
        assert results["pair_path"] == self.pair_path
        assert results["img_color"].shape == self.img_shape_gray
        assert results["img_color_ori_shape"] == self.img_shape_gray
        np.testing.assert_equal(results["img_color"], results["ori_img_color"])
        assert id(results["ori_img_color"]) != id(results["img_color"])
        assert results["img_color_path"] == self.pair_path
        assert results["img_gray"].shape == self.img_shape_gray
        assert results["img_gray_ori_shape"] == self.img_shape_gray
        np.testing.assert_equal(results["img_gray"], results["ori_img_gray"])
        assert id(results["ori_img_gray"]) != id(results["img_gray"])
        assert results["img_gray_path"] == self.pair_path
