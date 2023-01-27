"""Manipulate prediction box.

shape: [..., M], where M = 5 + N_CLASS e.g. M = 85 when N_CLASS = 80

format: (x, y, w, h, conf_logit, class_logit_1, class_logit_2, ...)
"""
import numpy as np
import tensorflow as tf

from .. import cfg
from ..types import Tensor
from ..types import TensorArr

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
    xy_raw = xy(y)
    wh_exp = wh(y)
    class_logit = class_logits(y)

    act_xy = (tf.sigmoid(xy_raw) + topleft_coords) * stride
    act_wh = (tf.exp(wh_exp) * anchors) * stride
    class_ids = tf.cast(tf.argmax(class_logit, axis=-1),
                        dtype=tf.float32)[..., tf.newaxis]
    return tf.concat(
        [act_xy, act_wh, conf(y), class_ids, class_logit],
        axis=-1,
    )
