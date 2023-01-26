"""Provide common types."""
from typing import TypeAlias

import numpy as np
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor

ALPHA = 0.1

TensorArr: TypeAlias = RaggedTensor | EagerTensor | np.ndarray
Tensor: TypeAlias = EagerTensor | RaggedTensor
