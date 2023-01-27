"""Manipulate bounding boxes.

shape: [..., M], where M >= 6:
    - M = 6 for ground truth label;
    - M > 6 for model prediction with class logit e.g. M = 86 if N_CLASS = 80

format: (x, y, w, h, conf, classid, [logit_1, logit_2, ...])
    - ground truth label: (x, y, w, h, 1, classid)
    - TRANSFORMED prediction:
        (x, y, w, h, conf_logit, class_id, class_logit_1, class_logit_2, ...)

A raw model prediction is NOT a bounding box;
transform it using functions in `pbox`
"""
from typing import Iterable

import tensorflow as tf

from .. import cfg
from ..types import TensorArr
from . import dbox


def objects(seq_label: Iterable[TensorArr]) -> tf.Tensor:
    """Get bounding boxes only with valid class IDs."""
    seq_bbox = [tf.reshape(label, (-1, 6)) for label in seq_label]
    bboxes = tf.concat(seq_bbox, axis=0)
    # get indices where class ID <> 0
    idx = tf.where(bboxes[..., 5])
    return tf.squeeze(tf.gather(bboxes, idx))


def xy(bbox: TensorArr) -> TensorArr:
    """From bounding box get coordinates tensor / array of shape (..., 2).

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., :2]


def wh(bbox: TensorArr) -> TensorArr:
    """From bbox get width and hieght tensor / array of shape (..., 2).

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., 2:4]


def xywh(bbox: TensorArr) -> TensorArr:
    """Get x-y coordinatestensor, width and height from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., :4]


def conf(bbox: TensorArr) -> TensorArr:
    """Get object confidence from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., 4:5]


def class_id(bbox: TensorArr, *, squeezed: bool = False) -> TensorArr:
    """Get class ID from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
        squeezed (bool): last dimension is 1 if False, squeezed otherwise
    """
    if squeezed:
        return bbox[..., 5]
    return bbox[..., 5:6]


def class_logits(bbox: TensorArr) -> TensorArr:
    """Get class logits from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., 6:]


def area(bbox: TensorArr) -> TensorArr:
    """Calculate area of a bounding box."""
    return bbox[..., 2] * bbox[..., 3]


def asdbox(bbox: TensorArr) -> TensorArr:
    """Transform a bounding box into a diagonal box."""
    return tf.concat(
        [
            xy(bbox) - wh(bbox) * 0.5,
            xy(bbox) + wh(bbox) * 0.5,
            bbox[..., 4:],
        ],
        axis=-1,
    )


def interarea(bbox_pred: TensorArr, bbox_label: TensorArr) -> TensorArr:
    """Intersection area of two bounding boxes."""
    dbox_pred = asdbox(bbox_pred)
    dbox_label = asdbox(bbox_label)
    return dbox.interarea(dbox_pred, dbox_label)


def iou(bbox_pred: TensorArr, bbox_label: TensorArr) -> TensorArr:
    """Calculate IoU of two bounding boxes."""
    area_pred = area(bbox_pred)
    area_label = area(bbox_label)
    area_inter = interarea(bbox_pred, bbox_label)
    area_union = area_pred + area_label - area_inter
    return (area_inter + cfg.EPSILON) / (area_union + cfg.EPSILON)
