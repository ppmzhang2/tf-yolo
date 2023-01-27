"""Manipulate width-height boxes."""
import numpy as np

from .. import cfg
from ..types import TensorArr

__all__ = ["iou"]


def iou(whs1: TensorArr, whs2: TensorArr) -> TensorArr:
    """IoU calculated without x and y, by aligning boxes along two edges.

    Args:
        whs1 (TensorArr): width and height of the first bounding boxes
        whs2 (TensorArr): width and height of the second bounding boxes

    Returns:
        TensorArr: Intersection over union of the corresponding boxes
    """
    intersection = (np.minimum(whs1[..., 0], whs2[..., 0]) *
                    np.minimum(whs1[..., 1], whs2[..., 1]))
    union = (whs1[..., 0] * whs1[..., 1] + whs2[..., 0] * whs2[..., 1] -
             intersection)
    return (intersection + cfg.EPSILON) / (union + cfg.EPSILON)
