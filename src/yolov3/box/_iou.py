import numpy as np
import tensorflow as tf

__all__ = ['iou_width_height', 'iou_bbox']


def iou_width_height(whs1, whs2):
    """IOU calculated without x and y,
    by aligning boxes along two edges

    Args:
        whs1 (TTensor): width and height of the first bounding boxes
        whs2 (TTensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = (np.minimum(whs1[..., 0], whs2[..., 0]) *
                    np.minimum(whs1[..., 1], whs2[..., 1]))
    union = (whs1[..., 0] * whs1[..., 1] + whs2[..., 0] * whs2[..., 1] -
             intersection)
    return intersection / union


def interarea_dbox(dbox_pred, dbox_label):
    """get IOU of diagonal boxes (containing two points on a diagonal,
    representing left-up and right-down points respectively)
    """

    def _right_down(dbox):
        return dbox[..., 2:]

    def _left_up(dbox):
        return dbox[..., :2]

    left_ups = tf.maximum(_left_up(dbox_pred), _left_up(dbox_label))
    right_downs = tf.minimum(_right_down(dbox_pred), _right_down(dbox_label))

    inter = tf.maximum(tf.subtract(right_downs, left_ups), 0.0)
    return tf.multiply(inter[..., 0], inter[..., 1])


def iou_bbox(dbox_pred, dbox_label):
    areas1 = dbox_pred[..., 2] * dbox_pred[..., 3]
    areas2 = dbox_label[..., 2] * dbox_label[..., 3]

    dbox_pred = tf.concat(
        [
            dbox_pred[..., :2] - dbox_pred[..., 2:] * 0.5,
            dbox_pred[..., :2] + dbox_pred[..., 2:] * 0.5
        ],
        axis=-1,
    )
    dbox_label = tf.concat(
        [
            dbox_label[..., :2] - dbox_label[..., 2:] * 0.5,
            dbox_label[..., :2] + dbox_label[..., 2:] * 0.5
        ],
        axis=-1,
    )

    inter = interarea_dbox(dbox_pred, dbox_label)
    union = areas1 + areas2 - inter

    return 1.0 * inter / union
