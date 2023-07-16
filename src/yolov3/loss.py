"""Training and loss."""
import logging
from collections.abc import Iterable
from typing import NoReturn

import numpy as np
import tensorflow as tf

from yolov3 import cfg
from yolov3.box import bbox
from yolov3.box import pbox
from yolov3.datasets.utils import onehot_cate_sn
from yolov3.types import Tensor

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
    lambda_obj: float = 10.0,
    lambda_bgd: float = 1.0,
) -> Tensor:
    """Calculate loss."""
    pred_ = pbox.scaled_bbox(pred)

    indices_obj = tf.where(bbox.conf(label, squeezed=True))
    indices_bgd = tf.where(bbox.conf(label, squeezed=True) == 1)

    prd_obj = tf.gather_nd(pred_, indices_obj)
    prd_bgd = tf.gather_nd(pred_, indices_bgd)
    lab_obj = tf.gather_nd(label, indices_obj)
    lab_bgd = tf.gather_nd(label, indices_bgd)

    ious = tf.expand_dims(bbox.iou(prd_obj, lab_obj), axis=-1)
    # background loss
    # TBD: weight with confidence
    loss_bgd = lambda_bgd * tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=bbox.conf(lab_bgd),
            logits=bbox.conf(prd_bgd),
        ))

    # object loss
    # label probability should be `1 * IOU` score according to the YOLO paper
    # TBD: weight with confidence
    loss_obj = lambda_obj * tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true=ious * bbox.conf(lab_obj),
            y_pred=bbox.conf(prd_obj),
            from_logits=True,
        ))

    # object center coordinates (xy) loss
    loss_xy = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(
            bbox.xy_offset(lab_obj),
            bbox.xy_offset(prd_obj),
        ))

    # box size (wh) loss
    loss_wh = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(
            bbox.wh_exp(lab_obj),
            bbox.wh_exp(prd_obj),
        ))

    # class loss
    loss_class = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true=onehot_cate_sn(bbox.class_sn(lab_obj, squeezed=True),
                                  N_CLASS),
            y_pred=bbox.class_logits(prd_obj),
            from_logits=True,
        ))

    LOGGER.info("\n"
                f"    Background Loss={loss_bgd};\n"
                f"    Object Loss={loss_obj};\n"
                f"    XY Loss={loss_xy};\n"
                f"    WH Loss={loss_wh};\n"
                f"    Class Loss={loss_class}")
    return loss_bgd + loss_obj + loss_xy + loss_wh + loss_class


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

        # TODO: handle loss_sum = nan
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
