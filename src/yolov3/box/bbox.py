"""Manipulate bounding boxes.

shape: [..., M], where M >= 10:
    - M = 10 for ground truth label;
    - M > 10 for model prediction with class logit e.g. M = 90 if N_CLASS = 80

format: (x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf, class SN,
         [logit_1, logit_2, ...])
    - ground truth label:
        (x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf_flag, class SN)
    - TRANSFORMED prediction:
        (x, y, w, h, x_offset, y_offset, w_exp, h_exp, conf_logit, class SN,
         class_logit_1, class_logit_2, ...)

A raw model prediction is NOT a bounding box;
transform it using functions in `pbox`
"""
import math

import tensorflow as tf

from .. import cfg
from ..types import Tensor
from ..types import TensorArr
from . import dbox


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


def xy_offset(bbox: TensorArr) -> TensorArr:
    """From bounding box get center poiont coordinates offset.

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., 4:6]


def wh_exp(bbox: TensorArr) -> TensorArr:
    """From bbox get width and height exponent of shape (..., 2).

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., 6:8]


def conf(bbox: TensorArr, *, squeezed: bool = False) -> TensorArr:
    """Get object confidence from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
        squeezed (bool): suppose the number of ranks of the input tensor is R,
            the #rank of the output tensor will be R - 1 is `squeezed = True`.
            Otherwise the #rank of the output will remain as R, and the last
            rank contains only 1 dimension
    """
    if squeezed:
        return bbox[..., 8]
    return bbox[..., 8:9]


def class_sn(bbox: TensorArr, *, squeezed: bool = False) -> TensorArr:
    """Get class ID from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
        squeezed (bool): suppose the number of ranks of the input tensor is R,
            the #rank of the output tensor will be R - 1 is `squeezed = True`.
            Otherwise the #rank of the output will remain as R, and the last
            rank contains only 1 dimension
    """
    if squeezed:
        return bbox[..., 9]
    return bbox[..., 9:10]


def class_logits(bbox: TensorArr) -> TensorArr:
    """Get class logits from a tensor / array.

    Args:
        bbox (TensorArr): bounding box
    """
    return bbox[..., 10:]


def area(bbox: TensorArr) -> TensorArr:
    """Calculate area of a bounding box."""
    return bbox[..., 2] * bbox[..., 3]


def asdbox(bbox: TensorArr) -> TensorArr:
    """Transform a bounding box into a diagonal box."""
    return tf.concat(
        [
            xy(bbox) - wh(bbox) * 0.5,
            xy(bbox) + wh(bbox) * 0.5,
            bbox[..., 8:],
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


def _inverse_sig(x: float) -> float:
    return math.log(x / (1 - x))


def objects(
    bbox: TensorArr,
    *,
    from_logits: bool = False,
    conf_th: float = 0.1,
) -> Tensor:
    """Get bounding boxes only with high confidence scores.

    Args:
        bbox (TensorArr): any bounding box of any valid shape
        from_logits (bool): whether the confidence score is logit or
            probability, default False
        conf_th (float): confidence threshold, default 0.1

    Return:
        Tensor: tensor of shape [N, 10] where N is the number of boxes
        containing an object, filtered by class ID
    """
    # get indices where class ID <> 0
    if from_logits:
        conf_th = _inverse_sig(conf_th)
    idx = tf.where(conf(bbox, squeezed=True) >= conf_th)
    return tf.gather_nd(bbox, idx)


def classof(bbox: TensorArr, sn: int) -> Tensor:
    """Filter a bbox by class SN."""
    idx = tf.where(class_sn(bbox, squeezed=True) == sn)
    return tf.gather_nd(bbox, idx)
