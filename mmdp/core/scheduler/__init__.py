# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks.lr_updater import OneCycleLrUpdaterHook

from .lr_updater import LinearLrUpdaterHook

__all__ = ["LinearLrUpdaterHook", "OneCycleLrUpdaterHook"]
