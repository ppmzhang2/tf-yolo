"""Training and loss."""
import logging
from collections.abc import Iterable
from typing import NoReturn

import numpy as np
import tensorflow as tf

from . import cfg
from .box import bbox
from .box import pbox
from .types import Tensor

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


def get_loss(
    pred: Tensor,
    label: Tensor,
    iou_threshold: float = 0.3,
) -> Tensor:
    """Calculate loss.

    TODO: add lambda coef
    """

    def calc(loss: Tensor) -> Tensor:
        return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2, 3, 4]))

    pred_bbox = pbox.scaled_bbox(pred)

    pred_xywh = bbox.xywh(pred_bbox)
    pred_conf_lg = bbox.conf(pred_bbox)
    pred_class_lg = bbox.class_logits(pred_bbox)

    label_xywh = bbox.xywh(label)
    label_conf_pr = bbox.conf(label)
    label_class = bbox.class_id(label, squeezed=True)
    label_class_pr = tf.one_hot(tf.cast(label_class, dtype=tf.int32), N_CLASS)

    iou_score = tf.expand_dims(bbox.iou(pred_xywh, label_xywh), axis=-1)
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


def grad(
    model: tf.keras.Model,
    x: Tensor,
    labels: Iterable[Tensor],
) -> tuple[float, Tensor]:
    """Get loss and gradients.

    TODO: update learning rate

    Args:
        model (tf.keras.Model): model for training
        x (Tensor): input features
        labels: (label_s, label_m, label_l)

    Return:
        tuple[float, Tensor]: total loss and gradient tensor
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


def trainer(model: tf.keras.Model, dataset: Iterable,
            n_epoch: int) -> NoReturn:
    """Model training loop."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for _ in range(n_epoch):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x, labels in dataset:
            loss, grads = grad(model, x, labels)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables, strict=True))
            epoch_loss_avg.update_state(loss)
            LOGGER.info(f"avg loss: {epoch_loss_avg}; loss: {loss}")
