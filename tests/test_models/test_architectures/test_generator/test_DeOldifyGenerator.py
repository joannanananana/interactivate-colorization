# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdp.models.architectures.generator import DeOldifyGenerator


class TestDeOldifyGenerator:
    @classmethod
    def setup_class(cls):
        cls.default_cfg = dict(
            type="DeOldifyGenerator",
            encoder=dict(
                type="ColorizationResNet",
                num_layers=101,
                pretrained=None,
                out_layers=[2, 5, 6, 7],
            ),
            mid_layers=dict(
                type="MidConvLayer", norm_type="NormSpectral", base_channels=2048
            ),
            decoder=dict(
                type="UnetWideDecoder",
                self_attention=True,
                x_in_c_list=[64, 256, 512, 1024],
                ni=2048,
                nf_factor=2,
                norm_type="NormSpectral",
            ),
            post_layers=dict(
                type="PostLayer",
                ni=256,
                last_cross=True,
                n_classes=3,
                bottle=False,
                norm_type="NormSpectral",
                y_range=(-3.0, 3.0),
            ),
        )

    def test_deoldify_generator_cpu(self):
        # test with default cfg
        real_a = torch.randn((2, 3, 256, 256))
        gen = DeOldifyGenerator(**self.default_cfg)
        fake_b = gen(real_a)
        assert fake_b.shape == (2, 3, 256, 256)
