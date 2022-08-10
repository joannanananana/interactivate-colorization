# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmdp.models.architectures.modules import (
    Flatten,
    custom_conv_layer,
    generation_init_weights,
)
from mmdp.models.builder import MODULES
from mmdp.utils import get_root_logger


@MODULES.register_module()
class DeoldifyDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 256,
        num_blocks: int = 3,
        p: int = 0.15,
        init_cfg=dict(type="normal", gain=0.02),
    ):
        super().__init__()

        self.init_type = (
            "normal" if init_cfg is None else init_cfg.get("type", "normal")
        )
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get("gain", 0.02)

        layers = [
            custom_conv_layer(
                in_channels,
                base_channels,
                ks=4,
                stride=2,
                leaky=0.2,
                norm_type="NormSpectral",
            ),
            nn.Dropout2d(p / 2),
        ]

        for i in range(num_blocks):
            layers += [
                custom_conv_layer(
                    base_channels,
                    base_channels,
                    ks=3,
                    stride=1,
                    leaky=0.2,
                    norm_type="NormSpectral",
                ),
                nn.Dropout2d(p / 2),
                custom_conv_layer(
                    base_channels,
                    base_channels * 2,
                    ks=4,
                    stride=2,
                    self_attention=(i == 0),
                    leaky=0.2,
                    norm_type="NormSpectral",
                ),
            ]
            base_channels = base_channels * 2

        layers += [
            custom_conv_layer(
                base_channels,
                base_channels,
                ks=3,
                stride=1,
                leaky=0.2,
                norm_type="NormSpectral",
            ),
            custom_conv_layer(
                base_channels,
                1,
                ks=4,
                bias=False,
                padding=0,
                use_activ=False,
                leaky=0.2,
                norm_type="NormSpectral",
            ),
            Flatten(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def init_weights(self, pretrained=None):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            generation_init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain
            )
        else:
            raise TypeError(
                "'pretrained' must be a str or None. "
                f"But received {type(pretrained)}."
            )
