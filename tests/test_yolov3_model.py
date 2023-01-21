from dataclasses import dataclass
from typing import Tuple

import pytest
import tensorflow as tf

from yolov3.model import TTensor
from yolov3.model import netv3

N = 2
N_CLASS = 80
N_BBOX = 5


@dataclass(frozen=True)
class Data:
    """input dataset and output shapes"""
    x: TTensor
    shape_s: Tuple[int, int, int, int, int]
    shape_m: Tuple[int, int, int, int, int]
    shape_l: Tuple[int, int, int, int, int]


in_shape = (N, 416, 416, 3)
x = tf.random.normal(in_shape)

dataset = [
    Data(x=x,
         shape_s=(N, 52, 52, 3, N_CLASS + N_BBOX),
         shape_m=(N, 26, 26, 3, N_CLASS + N_BBOX),
         shape_l=(N, 13, 13, 3, N_CLASS + N_BBOX))
]


@pytest.mark.parametrize('data', dataset)
def test_netv3(data: Data):
    t1, t2, t3 = netv3(data.x, N_CLASS)
    assert t1.shape == data.shape_s
    assert t2.shape == data.shape_m
    assert t3.shape == data.shape_l
