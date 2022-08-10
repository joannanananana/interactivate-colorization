import numpy as np

from mmdp.datasets.pipelines.Lq import (
    Lq_degradation_bsrgan,
    Lq_degradation_bsrgan_plus,
    Lq_downsample,
    Lq_util,
)


def test_Lq_util():
    img_gt = np.random.rand(256, 128, 3).astype(np.float32)
    results = dict(lq=img_gt, sf=4)

    assert results["lq"].shape == (256, 128, 3)
    assert results["sf"] == 4

    obj = Lq_util(key="lq", sf=4)
    results = obj(results)

    assert results["lq"].shape == (64, 32, 3)


def test_Lq_downsample():
    img_gt = np.random.rand(256, 128, 3).astype(np.float32)
    results = dict(lq=img_gt, sf=4)

    assert results["lq"].shape == (256, 128, 3)
    assert results["sf"] == 4

    obj = Lq_downsample(key="lq", sf=4)
    results = obj(results)

    assert results["lq"].shape == (64, 32, 3)


def test_Lq_degradation_bsrgan():
    img_gt = np.random.rand(256, 128, 3).astype(np.float32)
    results = dict(lq=img_gt, sf=4)
    assert results["lq"].shape == (256, 128, 3)
    assert results["sf"] == 4
    obj = Lq_degradation_bsrgan(key="lq", sf=4)
    results = obj(results)
    assert results["lq"].shape == (64, 32, 3)

    img_gt = np.random.rand(512, 512, 3).astype(np.float32)
    results = dict(lq=img_gt, sf=4)

    obj = Lq_degradation_bsrgan(key="lq", sf=4)
    results = obj(results)
    assert results["lq"].shape == (128, 128, 3)


def test_Lq_degradation_bsrgan_plus():
    img_gt = np.random.rand(512, 512, 3).astype(np.float32)

    results = dict(lq=img_gt, sf=4)

    assert results["lq"].shape == (512, 512, 3)
    assert results["sf"] == 4

    obj = Lq_degradation_bsrgan_plus(key="lq", sf=4)
    results = obj(results)

    assert results["lq"].shape == (128, 128, 3)

    img_gt_1 = np.random.rand(256, 128, 3).astype(np.float32)
    results = dict(lq=img_gt_1, sf=4)
    obj1 = Lq_degradation_bsrgan_plus(key="lq", sf=4)
    results = obj1(results)

    assert results["lq"].shape == (64, 32, 3)
