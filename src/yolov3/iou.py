import numpy as np


def iou_width_height(boxes1, boxes2):
    """IOU calculated without x and y,
    by aligning boxes along two edges

    Args:
        boxes1 (TTensor): width and height of the first bounding boxes
        boxes2 (TTensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = (np.minimum(boxes1[..., 0], boxes2[..., 0]) *
                    np.minimum(boxes1[..., 1], boxes2[..., 1]))
    union = (boxes1[..., 0] * boxes1[..., 1] +
             boxes2[..., 0] * boxes2[..., 1] - intersection)
    return intersection / union
