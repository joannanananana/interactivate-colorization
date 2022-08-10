# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmdp.core.evaluation.metrics import FID, IS, GaussianKLD


class TestFID:
    @classmethod
    def setup_class(cls):
        cls.reals = [torch.randn(2, 3, 128, 128) for _ in range(5)]
        cls.fakes = [torch.randn(2, 3, 128, 128) for _ in range(5)]

    def test_fid(self):
        fid = FID(
            3, inception_args=dict(normalize_input=False, load_fid_inception=False)
        )
        for b in self.reals:
            fid.feed(b, "reals")

        for b in self.fakes:
            fid.feed(b, "fakes")

        fid_score, mean, cov = fid.summary()
        assert fid_score > 0 and mean > 0 and cov > 0

        # To reduce the size of git repo, we remove the following test
        # fid = FID(
        #     3,
        #     inception_args=dict(
        #         normalize_input=False, load_fid_inception=False),
        #     inception_pkl=osp.join(
        #         osp.dirname(__file__), '..', 'data', 'test_dirty.pkl'))
        # assert fid.num_real_feeded == 3
        # for b in self.reals:
        #     fid.feed(b, 'reals')

        # for b in self.fakes:
        #     fid.feed(b, 'fakes')

        # fid_score, mean, cov = fid.summary()
        # assert fid_score > 0 and mean > 0 and cov > 0


class TestIS:
    @classmethod
    def setup_class(cls):
        cls.reals = [torch.randn(2, 3, 128, 128) for _ in range(5)]
        cls.fakes = [torch.randn(2, 3, 128, 128) for _ in range(5)]

    def test_is_cpu(self):
        inception_score = IS(10, resize=True, splits=10)
        inception_score.prepare()
        for b in self.reals:
            inception_score.feed(b, "reals")

        # pytorch version,downgrading the PyTorch version to 1.7
        # for b in self.fakes:
        #    inception_score.feed(b, 'fakes')

        # score, std = inception_score.summary()
        # assert score > 0 and std >= 0

    @torch.no_grad()
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_is_cuda(self):
        inception_score = IS(10, resize=True, splits=10)
        inception_score.prepare()
        for b in self.reals:
            inception_score.feed(b.cuda(), "reals")

        # pytorch version,downgrading the PyTorch version to 1.7
        # for b in self.fakes:
        #    inception_score.feed(b.cuda(), 'fakes')

        # score, std = inception_score.summary()
        # assert score > 0 and std >= 0


def test_kld_gaussian():
    # we only test at bz = 1 to test the numerical accuracy
    # due to the time and memory cost
    tar_shape = [2, 3, 4, 4]
    mean1, mean2 = torch.rand(*tar_shape, 1), torch.rand(*tar_shape, 1)
    # var1, var2 = torch.rand(2, 3, 4, 4, 1), torch.rand(2, 3, 4, 4, 1)
    var1 = torch.randint(1, 3, (*tar_shape, 1)).float()
    var2 = torch.randint(1, 3, (*tar_shape, 1)).float()

    def pdf(x, mean, var):
        return 1 / np.sqrt(2 * np.pi * var) * torch.exp(-((x - mean) ** 2) / (2 * var))

    delta = 0.0001
    indice = torch.arange(-5, 5, delta).repeat(*mean1.shape)
    p = pdf(indice, mean1, var1)  # pdf of target distribution
    q = pdf(indice, mean2, var2)  # pdf of predicted distribution

    kld_manually = (p * torch.log(p / q) * delta).sum(dim=(1, 2, 3, 4)).mean()

    data = dict(
        mean_pred=mean2,
        mean_target=mean1,
        logvar_pred=torch.log(var2),
        logvar_target=torch.log(var1),
    )

    metric = GaussianKLD(2)
    metric.prepare()
    metric.feed(data, "reals")
    kld = metric.summary()
    # this is a quite loose limitation for we cannot choose delta which is
    # small enough for precise kld calculation
    np.testing.assert_almost_equal(kld, kld_manually, decimal=1)
    # assert (kld - kld_manually < 1e-1).all()

    metric_base_2 = GaussianKLD(2, base="2")
    metric_base_2.prepare()
    metric_base_2.feed(data, "reals")
    kld_base_2 = metric_base_2.summary()
    np.testing.assert_almost_equal(kld_base_2, kld / np.log(2), decimal=4)
    # assert kld_base_2 == kld / np.log(2)

    # test wrong log_base
    with pytest.raises(AssertionError):
        GaussianKLD(2, base="10")

    # test other reduction --> mean
    metric = GaussianKLD(2, reduction="mean")
    metric.prepare()
    metric.feed(data, "reals")
    kld = metric.summary()

    # test other reduction --> sum
    metric = GaussianKLD(2, reduction="sum")
    metric.prepare()
    metric.feed(data, "reals")
    kld = metric.summary()

    # test other reduction --> error
    with pytest.raises(AssertionError):
        metric = GaussianKLD(2, reduction="none")
