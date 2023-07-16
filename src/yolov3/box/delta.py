"""Manipulate model raw output delta boxes.

shape: [..., 4 + K], where K is the number of classes:

format: (dx, dy, dw, dh, logit_1, logit_2, ...)
"""
from yolov3.types import TensorT


def dx(delta: TensorT) -> TensorT:
    """Get delta x coordinate of each delta box.

    Args:
        delta (TensorT): delta tensor of shape (H, W, 9, 4 + K)

    Returns:
        TensorT: delta tensor of shape (H, W, 9)
    """
    return delta[..., 0]


def dy(delta: TensorT) -> TensorT:
    """Get delta y coordinate of each delta box.

    Args:
        delta (TensorT): delta tensor of shape (H, W, 9, 4 + K)

    Returns:
        TensorT: delta tensor of shape (H, W, 9)
    """
    return delta[..., 1]


def dw(delta: TensorT) -> TensorT:
    """Get delta width of each delta box.

    Args:
        delta (TensorT): delta tensor of shape (H, W, 9, 4 + K)

    Returns:
        TensorT: delta tensor of shape (H, W, 9)
    """
    return delta[..., 2]


def dh(delta: TensorT) -> TensorT:
    """Get delta height of each delta box.

    Args:
        delta (TensorT): delta tensor of shape (H, W, 9, 4 + K)

    Returns:
        TensorT: delta tensor of shape (H, W, 9)
    """
    return delta[..., 3]


def logits(delta: TensorT) -> TensorT:
    """Get logits of each delta box.

    Args:
        delta (TensorT): delta tensor of shape (H, W, 9, 4 + K)

    Returns:
        TensorT: delta tensor of shape (H, W, 9, K)
    """
    return delta[..., 4:]
