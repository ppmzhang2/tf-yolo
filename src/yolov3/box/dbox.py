"""Manipulate diagonal boxes.

shape: [..., M], where M >= 6:
    - M = 6 for ground truth label;
    - M > 6 for model prediction with class logit e.g. M = 86 if N_CLASS = 80

format: (x_min, y_min, x_max, y_max, [conf], classid, [logit_1, logit_2, ...])
"""
import cv2
import numpy as np
import tensorflow as tf

from ..types import TensorArr

BOX_COLOR = (255, 0, 0)  # Red
BOX_THICKNESS = 1  # an integer
BOX_FONTSCALE = 0.35  # a float
TXT_COLOR = (255, 255, 255)  # White
TXT_THICKNESS = 1
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 0.35

CATE_MAP = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def pmax(dbox: TensorArr) -> TensorArr:
    """Get bottom-right point from a diagonal box."""
    return dbox[..., 2:4]


def pmin(dbox: TensorArr) -> TensorArr:
    """Get top-left point from a diagonal box."""
    return dbox[..., 0:2]


def interarea(dbox_pred: TensorArr, dbox_label: TensorArr) -> TensorArr:
    """Get intersection area of two Diagonal boxes."""
    left_ups = tf.maximum(pmin(dbox_pred), pmin(dbox_label))
    right_downs = tf.minimum(pmax(dbox_pred), pmax(dbox_label))

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
        x_min, y_min, x_max, y_max, _, cls_id = dbox
        class_name = CATE_MAP[int(cls_id)]
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
