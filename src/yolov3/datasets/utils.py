"""Dataset Util Functions."""
import tensorflow as tf

from ..types import Tensor

__all__ = ["onehot_cate_sn", "onecold_cate_sn"]


def onehot_cate_sn(sn: Tensor, n_class: int) -> Tensor:
    """Onehot coding for object category SN.

    Object category has three integer attributes: index, SN, ID. For COCO
    data (N_CLASS=80):
        - the ID can be well above 80, not suitable for `tf.one_hot`
        - index (ranges from 0 to 79) in ground truth cannot be used as 0
          is occupied to indicate background
        - to encode SN of range [1, 80], the onehot depth must be added to
          81, before removing the first dimension from the last rank of the
          output
    """
    return tf.one_hot(tf.cast(sn, dtype=tf.int32), n_class + 1)[..., 1:]


def onecold_cate_sn(logits: Tensor) -> Tensor:
    """Recover object category SN from category logits.

    Since the output range of `tf.argmax` is [0, N_CLASS - 1], `tf.ones` should
    be added to get the SN starting from 1.
    The output tensor will contain all the ranks of the input tensor except
    the last one.
    """
    outshape = logits.shape[:-1]
    return (tf.ones(outshape) +
            tf.cast(tf.argmax(logits, axis=-1), dtype=tf.float32))
