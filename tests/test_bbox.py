"""Test IoU functions of module box."""
from typing import NoReturn

import numpy as np
import tensorflow as tf

from yolov3.box import bbox

E = 1e-3
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


def test_bbox_objexts_pr() -> NoReturn:
    """Test bbox.objects."""
    tarr_exp = tf.constant(
        [
            [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.],
        ],
        dtype=tf.float64,
    )
    shape = (2, 13, 13, 3, 10)
    arr = np.zeros(shape)
    indices_label = [(0, 10, 6, 1, 8), (1, 5, 7, 2, 8)]
    for idx in indices_label:
        arr[idx] = 0.5
    res = bbox.objects(arr)
    assert abs((tarr_exp - res).numpy()).sum() < E


def test_bbox_objexts_all() -> NoReturn:
    """Test bbox.objects."""
    seq_pr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    arr_logit = tf.constant([
        [[0., 0., 0., 0., 0., 0., 0., 0., -1.38, 1.],
         [0., 0., 0., 0., 0., 0., 0., 0., -0.84, 2.]],
        [[0., 0., 0., 0., 0., 0., 0., 0., -0.40, 3.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.00, 4.]],
        [[0., 0., 0., 0., 0., 0., 0., 0., 0.41, 5.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.85, 6.]],
    ],
                            dtype=tf.float64)
    res = bbox.objects(arr_logit, from_logits=True, conf_th=seq_pr[0])
    assert res.shape == (6, 10)
    res = bbox.objects(arr_logit, from_logits=True, conf_th=seq_pr[1])
    assert res.shape == (5, 10)
    res = bbox.objects(arr_logit, from_logits=True, conf_th=seq_pr[2])
    assert res.shape == (4, 10)
    res = bbox.objects(arr_logit, from_logits=True, conf_th=seq_pr[3])
    assert res.shape == (3, 10)
    res = bbox.objects(arr_logit, from_logits=True, conf_th=seq_pr[4])
    assert res.shape == (2, 10)
    res = bbox.objects(arr_logit, from_logits=True, conf_th=seq_pr[5])
    assert res.shape == (1, 10)
