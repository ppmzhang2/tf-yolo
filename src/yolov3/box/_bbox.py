from typing import Iterable
from typing import TypeAlias
from typing import Union

import cv2
import numpy as np
import tensorflow as tf

from .. import db

TensorOrArr: TypeAlias = Union[tf.Tensor, np.ndarray]

BOX_COLOR = (255, 0, 0)  # Red
BOX_THICKNESS = 1  # an integer
BOX_FONTSCALE = 0.35  # a float
TXT_COLOR = (255, 255, 255)  # White
TXT_THICKNESS = 1
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 0.35

cate_map = db.dao.categories()


def label2bbox(label: TensorOrArr) -> TensorOrArr:
    """extract from label: x, y, w, h and class ID"""
    return label[..., 1:6]


def bbox_cleansing(seq_label: Iterable[TensorOrArr]) -> tf.Tensor:
    seq_bbox = [tf.reshape(label2bbox(label), (-1, 5)) for label in seq_label]
    bboxes = tf.concat(seq_bbox, axis=0)
    # get indices where class ID <> 0
    idx = tf.where(bboxes[..., 4])
    return tf.squeeze(tf.gather(bboxes, idx))


def bbox2coord(bbox: TensorOrArr) -> TensorOrArr:
    """from bbox to coordinates tensor / array of shape (..., 2)

    Args:
        bbox (TfArrayT): bounding box
    """
    return bbox[..., :2]


def bbox2wh(bbox: TensorOrArr) -> TensorOrArr:
    """from bbox to width and hieght tensor / array of shape (..., 2)

    Args:
        bbox (TfArrayT): bounding box
    """
    return bbox[..., 2:4]


def bbox2class(bbox: TensorOrArr) -> TensorOrArr:
    """from bbox to class ID tensor / array of shape (..., 1)

    Args:
        bbox (TfArrayT): bounding box
    """
    return bbox[..., 4:5]


def bbox2dbox(bbox: TensorOrArr) -> tf.Tensor:
    return tf.concat(
        [
            bbox2coord(bbox) - bbox2wh(bbox) * 0.5,
            bbox2coord(bbox) + bbox2wh(bbox) * 0.5,
            bbox2class(bbox),
        ],
        axis=-1,
    )


def img_add_box(img: np.ndarray, dboxes: TensorOrArr) -> np.ndarray:
    """add bounding boxes to an image array

    Args:
        img (np.ndarray): image NumPy array
        dboxes (TfArrayT): diagonal boxes array of shape (N_BOX, 5)

    Returns:
        np.ndarray: image NumPy array with bounding boxes added
    """
    for dbox in dboxes.numpy().astype(np.int32):
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
