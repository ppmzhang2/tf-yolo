from typing import Iterable

import cv2
import numpy as np
import tensorflow as tf

from .. import cfg
from .. import db
from ..types import TensorArr

__all__ = ["iou_bbox"]

BOX_COLOR = (255, 0, 0)  # Red
BOX_THICKNESS = 1  # an integer
BOX_FONTSCALE = 0.35  # a float
TXT_COLOR = (255, 255, 255)  # White
TXT_THICKNESS = 1
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 0.35

cate_map = db.dao.categories()


def label2bbox(label: TensorArr) -> TensorArr:
    """Label to bounding box.

    extract from label: x, y, w, h and class ID
    """
    return label[..., 1:6]


def bbox_cleansing(seq_label: Iterable[TensorArr]) -> tf.Tensor:
    """Bounding box cleansing."""
    seq_bbox = [tf.reshape(label2bbox(label), (-1, 5)) for label in seq_label]
    bboxes = tf.concat(seq_bbox, axis=0)
    # get indices where class ID <> 0
    idx = tf.where(bboxes[..., 4])
    return tf.squeeze(tf.gather(bboxes, idx))


def bbox_coord(bbox: TensorArr) -> TensorArr:
    """From bounding box get coordinates tensor / array of shape (..., 2).

    Args:
        bbox (TfArrayT): bounding box
    """
    return bbox[..., :2]


def bbox_wh(bbox: TensorArr) -> TensorArr:
    """From bbox get width and hieght tensor / array of shape (..., 2).

    Args:
        bbox (TfArrayT): bounding box
    """
    return bbox[..., 2:4]


def bbox_class(bbox: TensorArr) -> TensorArr:
    """From bbox get class ID tensor / array of shape (..., 1).

    Args:
        bbox (TfArrayT): bounding box
    """
    return bbox[..., 4:5]


def bbox_area(bbox: TensorArr) -> TensorArr:
    return bbox[..., 2] * bbox[..., 3]


def bbox2dbox(bbox: TensorArr) -> TensorArr:
    return tf.concat(
        [
            bbox_coord(bbox) - bbox_wh(bbox) * 0.5,
            bbox_coord(bbox) + bbox_wh(bbox) * 0.5,
            bbox[..., 4:],
        ],
        axis=-1,
    )


def dbox_pmax(dbox: TensorArr) -> TensorArr:
    """Get bottom-right point from a diagonal box."""
    return dbox[..., 2:4]


def dbox_pmin(dbox: TensorArr) -> TensorArr:
    """Get top-left point from a diagonal box."""
    return dbox[..., 0:2]


def dbox_inter(dbox_pred: TensorArr, dbox_label: TensorArr) -> TensorArr:
    """Get intersection area of two Diagonal boxes."""
    left_ups = tf.maximum(dbox_pmin(dbox_pred), dbox_pmin(dbox_label))
    right_downs = tf.minimum(dbox_pmax(dbox_pred), dbox_pmax(dbox_label))

    inter = tf.maximum(tf.subtract(right_downs, left_ups), 0.0)
    return tf.multiply(inter[..., 0], inter[..., 1])


def img_add_box(img: np.ndarray, dboxes: TensorArr) -> np.ndarray:
    """Add bounding boxes to an image array.

    Args:
        img (np.ndarray): image NumPy array
        dboxes (TfArrayT): diagonal boxes array of shape (N_BOX, 5)

    Returns:
        np.ndarray: image NumPy array with bounding boxes added
    """
    for dbox in (dboxes.astype(np.int32) if isinstance(dboxes, np.ndarray) else
                 dboxes.numpy().astype(np.int32)):
        x_min, y_min, x_max, y_max, cls_id = dbox
        class_name = cate_map[int(cls_id)]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=BOX_COLOR,
                      thickness=BOX_THICKNESS)

        (text_width, text_height), _ = cv2.getTextSize(class_name, FONTFACE,
                                                       FONTSCALE,
                                                       TXT_THICKNESS)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                      (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=FONTFACE,
            fontScale=FONTSCALE,
            color=TXT_COLOR,
            lineType=cv2.LINE_AA,
        )
    return img


def bbox_inter(bbox_pred: TensorArr, bbox_label: TensorArr) -> TensorArr:
    dbox_pred = bbox2dbox(bbox_pred)
    dbox_label = bbox2dbox(bbox_label)
    return dbox_inter(dbox_pred, dbox_label)


def iou_bbox(bbox_pred: TensorArr, bbox_label: TensorArr) -> TensorArr:
    area_pred = bbox_area(bbox_pred)
    area_label = bbox_area(bbox_label)
    inter = bbox_inter(bbox_pred, bbox_label)
    union = area_pred + area_label - inter
    return (inter + cfg.EPSILON) / (union + cfg.EPSILON)
