# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import pytest
from mmcv.utils.testing import assert_dict_has_keys

from mmdp.datasets import (
    BaseSRDataset,
    SRAnnotationDataset,
    SRFolderDataset,
    SRFolderGTDataset,
)


def mock_open(*args, **kwargs):
    """unittest.mock_open wrapper.

    unittest.mock_open doesn't support iteration. Wrap it to fix this bug.
    Reference: https://stackoverflow.com/a/41656192
    """
    import unittest

    f_open = unittest.mock.mock_open(*args, **kwargs)
    f_open.return_value.__iter__ = lambda self: iter(self.readline, "")
    return f_open


class TestSRDatasets:
    @classmethod
    def setup_class(cls):
        cls.data_prefix = Path(__file__).parent.parent.parent / "data"

    def test_base_super_resolution_dataset(self):
        class ToyDataset(BaseSRDataset):
            """Toy dataset for testing SRDataset."""

            def __init__(self, pipeline, test_mode=False):
                super().__init__(pipeline, test_mode)

            def load_annotations(self):
                pass

            def __len__(self):
                return 2

        toy_dataset = ToyDataset(pipeline=[])
        file_paths = [osp.join("gt", "baboon.png"), osp.join("lq", "baboon_x4.png")]
        file_paths = [str(self.data_prefix / v) for v in file_paths]

        result = toy_dataset.scan_folder(self.data_prefix)
        assert set(file_paths).issubset(set(result))
        result = toy_dataset.scan_folder(str(self.data_prefix))
        assert set(file_paths).issubset(set(result))

        with pytest.raises(TypeError):
            toy_dataset.scan_folder(123)

        # test evaluate function
        results = [
            {"eval_result": {"PSNR": 20, "SSIM": 0.6}},
            {"eval_result": {"PSNR": 30, "SSIM": 0.8}},
        ]

        with pytest.raises(TypeError):
            # results must be a list
            toy_dataset.evaluate(results=5)
        with pytest.raises(AssertionError):
            # The length of results should be equal to the dataset len
            toy_dataset.evaluate(results=[results[0]])

        eval_result = toy_dataset.evaluate(results=results)
        assert eval_result == {"PSNR": 25, "SSIM": 0.7}

        with pytest.raises(AssertionError):
            results = [
                {"eval_result": {"PSNR": 20, "SSIM": 0.6}},
                {"eval_result": {"PSNR": 30}},
            ]
            # Length of evaluation result should be the same as the dataset len
            toy_dataset.evaluate(results=results)

    def test_sr_annotation_dataset(self):
        # setup
        anno_file_path = self.data_prefix / "train.txt"
        sr_pipeline = [
            dict(type="LoadImageFromFile", io_backend="disk", key="lq"),
            dict(type="LoadImageFromFile", io_backend="disk", key="gt"),
            dict(type="PairedRandomCrop", gt_patch_size=128),
            dict(type="ImageToTensor", keys=["lq", "gt"]),
        ]
        target_keys = [
            "lq_path",
            "gt_path",
            "scale",
            "lq",
            "lq_ori_shape",
            "gt",
            "gt_ori_shape",
        ]

        # input path is Path object
        sr_annotation_dataset = SRAnnotationDataset(
            lq_folder=self.data_prefix / "lq",
            gt_folder=self.data_prefix / "gt",
            ann_file=anno_file_path,
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl="{}_x4",
        )
        data_infos = sr_annotation_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(self.data_prefix / "lq" / "baboon_x4.png"),
                gt_path=str(self.data_prefix / "gt" / "baboon.png"),
            )
        ]
        result = sr_annotation_dataset[0]
        assert len(sr_annotation_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)
        # input path is str
        sr_annotation_dataset = SRAnnotationDataset(
            lq_folder=str(self.data_prefix / "lq"),
            gt_folder=str(self.data_prefix / "gt"),
            ann_file=str(anno_file_path),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl="{}_x4",
        )
        data_infos = sr_annotation_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(self.data_prefix / "lq" / "baboon_x4.png"),
                gt_path=str(self.data_prefix / "gt" / "baboon.png"),
            )
        ]
        result = sr_annotation_dataset[0]
        assert len(sr_annotation_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)

    def test_sr_folder_dataset(self):
        # setup
        sr_pipeline = [
            dict(type="LoadImageFromFile", io_backend="disk", key="lq"),
            dict(type="LoadImageFromFile", io_backend="disk", key="gt"),
            dict(type="PairedRandomCrop", gt_patch_size=128),
            dict(type="ImageToTensor", keys=["lq", "gt"]),
        ]
        target_keys = ["lq_path", "gt_path", "scale", "lq", "gt"]
        lq_folder = self.data_prefix / "lq"
        gt_folder = self.data_prefix / "gt"
        filename_tmpl = "{}_x4"

        # input path is Path object
        sr_folder_dataset = SRFolderDataset(
            lq_folder=lq_folder,
            gt_folder=gt_folder,
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl,
        )
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(lq_folder / "baboon_x4.png"),
                gt_path=str(gt_folder / "baboon.png"),
            )
        ]
        result = sr_folder_dataset[0]
        assert len(sr_folder_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)
        # input path is str
        sr_folder_dataset = SRFolderDataset(
            lq_folder=str(lq_folder),
            gt_folder=str(gt_folder),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl,
        )
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [
            dict(
                lq_path=str(lq_folder / "baboon_x4.png"),
                gt_path=str(gt_folder / "baboon.png"),
            )
        ]
        result = sr_folder_dataset[0]
        assert len(sr_folder_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)

    def test_sr_folder_gt_dataset(self):
        # setup
        sr_pipeline = [
            dict(type="LoadImageFromFile", io_backend="disk", key="gt"),
            dict(type="ImageToTensor", keys=["gt"]),
        ]
        target_keys = ["gt_path", "gt"]
        gt_folder = self.data_prefix / "gt"
        filename_tmpl = "{}_x4"

        # input path is Path object
        sr_folder_dataset = SRFolderGTDataset(
            gt_folder=gt_folder,
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl,
        )
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [dict(gt_path=str(gt_folder / "baboon.png"))]
        result = sr_folder_dataset[0]
        assert len(sr_folder_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)
        # input path is str
        sr_folder_dataset = SRFolderGTDataset(
            gt_folder=str(gt_folder),
            pipeline=sr_pipeline,
            scale=4,
            filename_tmpl=filename_tmpl,
        )
        data_infos = sr_folder_dataset.data_infos
        assert data_infos == [dict(gt_path=str(gt_folder / "baboon.png"))]
        result = sr_folder_dataset[0]
        assert len(sr_folder_dataset) == 1
        assert assert_dict_has_keys(result, target_keys)
