import tensorflow as tf

from . import cfg
from .loss import get_loss
from .model import netv3

N_CHANNEL = 3


def model_factory():
    x = tf.keras.layers.Input([cfg.V3IN_WIDTH, cfg.V3IN_WIDTH, N_CHANNEL])
    seq_out = netv3(x)
    return tf.keras.Model(x, seq_out)


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


def trainer(dataset, num_epochs: int = 2):
    model = model_factory()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    for ep in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x, labels in dataset:
            loss, grads = grad(model, x, labels)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss)
            print(f"avg loss: {epoch_loss_avg}; loss: {loss}")
