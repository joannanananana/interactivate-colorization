# Copyright (c) OpenMMLab. All rights reserved.
from mmdp.apis.train import set_random_seed


def test_set_random_seed():
    set_random_seed(0, deterministic=False)
    set_random_seed(0, deterministic=True)
