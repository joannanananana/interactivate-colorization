import numpy as np
import pytest
import torch

from mmdp.models import (
    decode_max_ab,
    decode_mean,
    encode_ab_ind,
    get_colorization_data,
    lab2rgb,
)


def test_get_colorization():
    data_raw = torch.rand(1, 3, 256, 256)
    outputs = get_colorization_data(data_raw)
    assert isinstance(outputs, dict)
    assert outputs["A"].size() == (1, 1, 256, 256)
    with pytest.raises(TypeError):
        data_raw = -1 + 2 * torch.rand(1, 3, 256, 256)
        outputs = get_colorization_data(data_raw)
        outputs["A"].size()


def test_encode_ab_ind():
    opt = dict(
        ab_quant=10,
        ab_max=110,
        ab_norm=110,
        A=2 * 110 / 10 + 1,
        B=2 * 110 / 10 + 1,
        sample_p=1,
    )
    data_ab = -1 + 2 * torch.rand(1, 2, 256, 256)
    assert encode_ab_ind(data_ab, opt).size() == ((1, 1, 256, 256))


def test_decode_max_ab():
    opt = dict(
        ab_quant=10,
        ab_max=110,
        ab_norm=110,
        A=2 * 110 / 10 + 1,
        B=2 * 110 / 10 + 1,
        sample_p=1,
    )
    data_ab_quant = torch.rand(1, 3, 256, 256)
    outputs = decode_max_ab(data_ab_quant, opt)
    assert outputs.size() == (1, 2, 256, 256)
    index = np.unravel_index(outputs.argmax(), outputs.shape)
    assert outputs[index[0], index[1], index[2], index[3]] <= 1
    assert outputs.size() == (1, 2, 256, 256)
    index = np.unravel_index(outputs.argmin(), outputs.shape)
    assert outputs[index[0], index[1], index[2], index[3]] >= -1


def test_decode_mean():
    opt = dict(
        ab_quant=10,
        ab_max=110,
        ab_norm=110,
        A=2 * 110 / 10 + 1,
        B=2 * 110 / 10 + 1,
        sample_p=1,
    )
    data_ab_quant = torch.rand(1, 529, 256, 256)
    outputs = decode_mean(data_ab_quant, opt)
    assert outputs.size() == (1, 2, 256, 256)


def test_lab2rgb():
    opt = dict(
        A=1,
        l_norm=100,
        l_cent=50,
        ab_norm=110,
    )
    lab_rs = torch.rand(1, 3, 256, 256)
    outputs = lab2rgb(lab_rs, opt)
    assert outputs.size() == (1, 3, 256, 256)
