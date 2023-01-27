"""Test IoU functions of module box."""
from typing import NoReturn

import numpy as np
import tensorflow as tf

from yolov3.box import bbox

BBOX1 = (2.5, 3.4, 5.0, 6.0)
BBOX2 = (2.6, 4.3, 6.0, 4.0)
LMT_UP = 0.5883
LMT_LOW = 0.5882

bboxes1 = np.array((BBOX1, BBOX1))
bboxes2 = np.array((BBOX2, BBOX2))


def test_bbox_iou() -> NoReturn:
    """Test bbox.iou."""
    res = bbox.iou(bboxes1, bboxes2).numpy()
    assert np.all(res > LMT_LOW)
    assert np.all(res < LMT_UP)


def test_bbox_objexts() -> NoReturn:
    """Test bbox.objects."""
    shape = (2, 13, 13, 3, 6)
    arr = np.zeros(shape)
    indices_label = [(0, 9, 6, 1, 5), (1, 5, 7, 2, 5)]
    for idx in indices_label:
        arr[idx] = 1
    res = bbox.objects(arr)
    tarr_exp = tf.constant([
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.],
    ])
    assert res.shape == tarr_exp.shape
    assert res.numpy().sum() == tarr_exp.numpy().sum()
