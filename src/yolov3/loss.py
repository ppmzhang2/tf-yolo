import numpy as np
import tensorflow as tf

from . import cfg
from .iou import iou_bbox

N_CLASS = 80

STRIDE_MAP = {scale: cfg.V3IN_WIDTH // scale for scale in cfg.V3ANCHORSCALES}

# anchors measured in corresponding strides
ANCHORS_IN_STRIDE = (np.array(cfg.V3ANCHORS).T /
                     np.array(sorted(STRIDE_MAP.values()))).T

# map from grid size to anchors measured in stride
ANCHORS_MAP = {
    scale: ANCHORS_IN_STRIDE[i]
    for i, scale in enumerate(cfg.V3ANCHORSCALES)
}


def pred2act(y):
    """transform prediction to actual size"""
    grid_size = y.shape[1]
    stride = STRIDE_MAP[grid_size]
    anchors = ANCHORS_MAP[grid_size]
    offset_raw = y[..., :2]
    wh_exp = y[..., 2:4]
    conf_logit = y[..., 4:5]
    class_logit = y[..., 5:]

    act_offset = tf.sigmoid(offset_raw) * stride
    act_wh = (tf.exp(wh_exp) * anchors) * stride
    conf_pr = tf.sigmoid(conf_logit)
    # class_pr = tf.sigmoid(class_logit)
    return tf.concat(
        [act_offset, act_wh, conf_logit, conf_pr, class_logit],
        axis=-1,
    )


def get_loss(pred, label, iou_threshold=0.3):
    """
    TODO: add lambda coef
    """

    def calc(loss):
        return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))

    pred = pred2act(pred)

    pred_xywh = pred[..., 0:4]
    pred_conf_lg = pred[..., 4:5]
    pred_class_lg = pred[..., 6:]

    label_xywh = label[..., 0:4]
    label_conf_pr = label[..., 4:5]
    label_class = label[..., 5]
    label_class_pr = tf.one_hot(tf.cast(label_class, dtype=tf.int32), N_CLASS)

    iou_score = tf.expand_dims(iou_bbox(pred_xywh, label_xywh), axis=-1)
    noobj_pr = tf.cast(iou_score < iou_threshold, tf.float32)
    loss_iou = tf.multiply(label_conf_pr, tf.subtract(1, iou_score))
    loss_conf = tf.multiply(
        label_conf_pr,
        tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf_pr,
                                                logits=pred_conf_lg),
    ) + tf.multiply(
        noobj_pr,
        tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf_pr,
                                                logits=pred_conf_lg),
    )
    loss_class = tf.multiply(
        label_conf_pr,
        tf.nn.sigmoid_cross_entropy_with_logits(labels=label_class_pr,
                                                logits=pred_class_lg),
    )
    return calc(loss_iou) + calc(loss_conf) + calc(loss_class)
