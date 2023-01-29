"""Test mAP functions of module metrics."""
from typing import NoReturn

import numpy as np

from yolov3.box import bbox
from yolov3.metrics import _mean_ap

E = 1e-2


def test_ap_seq() -> NoReturn:
    """Test function `ap_seq`."""
    prd_xywh = np.array([
        [0.8, 0.7, 0.2, 0.2],
        [0.15, 0.25, 0.1, 0.1],
        [0.35, 0.6, 0.3, 0.2],
        [0.5, 0.43, 0.12, 0.75],
    ])

    arr_xywh = np.array([
        [0.55, 0.2, 0.3, 0.2],
        [0.35, 0.6, 0.3, 0.2],
        [0.8, 0.7, 0.2, 0.2],
    ])
    conf = np.array([0.67, 0.75, 0.23, 0.49])
    exp = np.array([0.0196, -1., 0.1968, 1.])

    arr_iou = np.apply_along_axis(lambda x: bbox.iou(x, arr_xywh), 1, prd_xywh)
    res = _mean_ap.ap_1img1class(arr_iou, conf)
    assert (exp - res).sum() < E
