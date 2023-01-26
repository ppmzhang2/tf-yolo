import logging
from collections.abc import Iterable

import numpy as np
import tensorflow as tf

from . import cfg
from . import box

LOGGER = logging.getLogger(__name__)

N_CLASS = 80

STRIDE_MAP = {scale: cfg.V3_INRESOLUT // scale for scale in cfg.V3_GRIDSIZE}

# anchors measured in corresponding strides
ANCHORS_IN_STRIDE = (np.array(cfg.V3_ANCHORS).T /
                     np.array(sorted(STRIDE_MAP.values()))).T

# map from grid size to anchors measured in stride
ANCHORS_MAP = {
    scale: ANCHORS_IN_STRIDE[i]
    for i, scale in enumerate(cfg.V3_GRIDSIZE)
}


def grid_coord(batch_size: int, grid_size: int, n_anchor: int):
    """top-left coordinates of each grid cells, created usually out of a output
    tensor

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


def pred2act(y):
    """transform prediction to actual size"""
    batch_size = y.shape[0]
    grid_size = y.shape[1]
    n_anchor = y.shape[3]
    stride = STRIDE_MAP[grid_size]
    anchors = ANCHORS_MAP[grid_size]
    topleft_coords = grid_coord(batch_size, grid_size, n_anchor)
    offset_raw = y[..., :2]
    wh_exp = y[..., 2:4]
    conf_logit = y[..., 4:5]
    class_logit = y[..., 5:]

    act_xy = (tf.sigmoid(offset_raw) + topleft_coords) * stride
    act_wh = (tf.exp(wh_exp) * anchors) * stride
    conf_pr = tf.sigmoid(conf_logit)
    # class_pr = tf.sigmoid(class_logit)
    return tf.concat(
        [act_xy, act_wh, conf_logit, conf_pr, class_logit],
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

    label_conf_pr = label[..., 0:1]
    label_xywh = label[..., 1:5]
    label_class = label[..., 5]
    label_class_pr = tf.one_hot(tf.cast(label_class, dtype=tf.int32), N_CLASS)

    iou_score = tf.expand_dims(box.iou_bbox(pred_xywh, label_xywh), axis=-1)
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
    LOGGER.info("    "
                f"IOU Loss={calc(loss_iou)}; "
                f"Conf Loss={calc(loss_conf)}; "
                f"Class Loss={calc(loss_class)}")
    return calc(loss_iou) + calc(loss_conf) + calc(loss_class)


def grad(model, x, labels):
    """
    TODO: update learning rate
    Args:
        labels: (label_s, label_m, label_l)
    """
    with tf.GradientTape() as tape:
        seq_pred = model(x)

        loss_sum = 0
        for i, pred in enumerate(seq_pred):
            loss = get_loss(pred, labels[i])
            loss_sum += loss

        gradients = tape.gradient(loss_sum, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_sum, gradients


def trainer(model: tf.keras.Model, dataset: Iterable, n_epoch: int):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for ep in range(n_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x, labels in dataset:
            loss, grads = grad(model, x, labels)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
            LOGGER.info(f"avg loss: {epoch_loss_avg}; loss: {loss}")
