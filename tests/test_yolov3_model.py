"""test Modles."""
from dataclasses import dataclass
from typing import NoReturn

import pytest
import tensorflow as tf

from yolov3.model import netv3
from yolov3.types import TensorArr

N = 2
N_CLASS = 80
N_BBOX = 5


@dataclass(frozen=True)
class Data:
    """Input dataset and output shapes."""
    x: TensorArr
    shape_s: tuple[int, int, int, int, int]
    shape_m: tuple[int, int, int, int, int]
    shape_l: tuple[int, int, int, int, int]


in_shape = (N, 416, 416, 3)
x = tf.random.normal(in_shape)

dataset = [
    Data(x=x,
         shape_s=(N, 52, 52, 3, N_CLASS + N_BBOX),
         shape_m=(N, 26, 26, 3, N_CLASS + N_BBOX),
         shape_l=(N, 13, 13, 3, N_CLASS + N_BBOX)),
]


@pytest.mark.parametrize("data", dataset)
def test_netv3(data: Data) -> NoReturn:
    """Test `netv3` YOLO model."""
    t1, t2, t3 = netv3(data.x, N_CLASS)
    assert t1.shape == data.shape_s
    assert t2.shape == data.shape_m
    assert t3.shape == data.shape_l
