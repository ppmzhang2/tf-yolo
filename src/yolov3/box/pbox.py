"""Manipulate prediction box.

shape: [..., M], where M = 5 + N_CLASS e.g. M = 85 when N_CLASS = 80

format: (x, y, w, h, conf_logit, class_logit_1, class_logit_2, ...)
"""
import numpy as np
import tensorflow as tf

from yolov3 import cfg
from yolov3.datasets.utils import onecold_cate_sn
from yolov3.types import Tensor
from yolov3.types import TensorArr

STRIDE_MAP = {scale: cfg.V3_INRESOLUT // scale for scale in cfg.V3_GRIDSIZE}

# anchors measured in corresponding strides
ANCHORS_IN_STRIDE = (np.array(cfg.V3_ANCHORS).T /
                     np.array(sorted(STRIDE_MAP.values()))).T

# map from grid size to anchors measured in stride
ANCHORS_MAP = {
    scale: ANCHORS_IN_STRIDE[i]
    for i, scale in enumerate(cfg.V3_GRIDSIZE)
}


def xy(pbox: TensorArr) -> TensorArr:
    """Get x-y coordinatestensor from a tensor / array.

    Args:
        pbox (TensorArr): model predicted box
    """
    return pbox[..., :2]


def wh(pbox: TensorArr) -> TensorArr:
    """Get width and height from a tensor / array.

    Args:
        pbox (TensorArr): model predicted box
    """
    return pbox[..., 2:4]


def xywh(pbox: TensorArr) -> TensorArr:
    """Get x-y coordinatestensor, width and height from a tensor / array.

    Args:
        pbox (TensorArr): bounding box
    """
    return pbox[..., :4]


def conf(pbox: TensorArr) -> TensorArr:
    """Get object confidence from a tensor / array.

    Args:
        pbox (TensorArr): model predicted box
    """
    return pbox[..., 4:5]


def class_logits(pbox: TensorArr) -> TensorArr:
    """Get class logits from a tensor / array.

    Args:
        pbox (TensorArr): model predicted box
    """
    return pbox[..., 5:]


def _grid_coord(batch_size: int, grid_size: int, n_anchor: int) -> Tensor:
    """Top-left coordinates of each grid cells.

    created usually out of a output tensor

    Args:
        batch_size (int): batch size
        grid_size (int): grid size, could be 13 (large), 26 (medium) or 52
            (small)
        n_anchor (int): number of anchors of a specific grid size, usually
            should be 3
    """
    vec = tf.range(0, limit=grid_size, dtype=tf.float32)  # x or y range
    xs = vec[tf.newaxis, :, tf.newaxis, tf.newaxis]
    ys = vec[tf.newaxis, tf.newaxis, :, tf.newaxis]
    xss = tf.tile(xs, (batch_size, 1, grid_size, n_anchor))
    yss = tf.tile(ys, (batch_size, grid_size, 1, n_anchor))
    return tf.stack([xss, yss], axis=-1)


def scaled_bbox(y: Tensor) -> Tensor:
    """Transform prediction to actual sized `bbox`."""
    batch_size = y.shape[0]
    grid_size = y.shape[1]  # 52, 26 or 13
    n_anchor = y.shape[3]  # 3
    stride = STRIDE_MAP[grid_size]
    anchors = ANCHORS_MAP[grid_size]
    topleft_coords = _grid_coord(batch_size, grid_size, n_anchor)
    xy_sig = tf.sigmoid(xy(y))
    wh_exp = tf.sigmoid(wh(y))
    cate_logit = class_logits(y)

    xy_act = (xy_sig + topleft_coords) * stride
    wh_act = (tf.exp(wh_exp) * anchors) * stride
    cate_sn = onecold_cate_sn(cate_logit)
    return tf.concat(
        [
            xy_act,
            wh_act,
            xy_sig,
            wh_exp,
            conf(y),
            cate_sn[..., tf.newaxis],
            cate_logit,
        ],
        axis=-1,
    )
