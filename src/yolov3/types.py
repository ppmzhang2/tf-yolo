"""Provide common types."""
from typing import TypeAlias
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor

TensorArr: TypeAlias = RaggedTensor | EagerTensor | np.ndarray
Tensor: TypeAlias = EagerTensor | RaggedTensor
TensorT = Union[tf.Tensor, np.ndarray]  # noqa: UP007
