# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmdp.models.architectures.modules import generation_init_weights
from mmdp.models.builder import MODULES, build_module
from mmdp.utils import get_root_logger


@MODULES.register_module()
class DeOldifyGenerator(nn.Module):
    def __init__(
        self,
        encoder,
        mid_layers,
        decoder,
        post_layers,
        init_cfg=dict(type="normal", gain=0.02),
        **kwags,
    ):
        super().__init__()

        self.layers_enc = build_module(encoder)
        self.layers_mid = build_module(mid_layers)
        self.layers_dec = build_module(decoder)
        self.layers_post = build_module(post_layers)

        self.init_type = (
            "normal" if init_cfg is None else init_cfg.get("type", "normal")
        )
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get("gain", 0.02)

    def forward(self, x):
        res = x
        res, short_cut_out = self.layers_enc(res)
        res = self.layers_mid(res)

        ## reverse the feature maps order in the list
        short_cut_out.reverse()
        res = self.layers_dec(res, short_cut_out)

        res = self.layers_post(res, x)
        return res

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
