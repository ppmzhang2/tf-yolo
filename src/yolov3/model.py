from typing import Tuple
from typing import Union

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.framework.sparse_tensor import SparseTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor

ALPHA = 0.1

TTensor = Union[Tensor, SparseTensor, RaggedTensor, EagerTensor]


def cnn_block(
    x: TTensor,
    filters: int,
    kernel_size: int,
    downsample: bool = False,
    bn: bool = True,
) -> TTensor:
    """Conv2D block
    Args:
        x (TTensor): input tensor
        filters (int): number of output channels
        kernel_size (int): kernel size, 1 or 3
        downsample (bool): down-sampling flag i.e. set stride to 2 if
            downsample add stride 1 otherwise.
            Always do the 0 padding i.e. padding = 'same'
        bn (bool): batch-normalization flag, True means:
            1. this layer is for prediction
            2. use batch-normalization
            3. Leaky-ReLU activation

    Returns:
        TTensor: output tensor
    """
    # if downsample add stride without padding,
    # otherwise stride 1 with zero padding i.e. keep the spatial dimensions
    if downsample:
        strides = 2
    else:
        strides = 1

    x_ = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        use_bias=not bn,
        padding='same',
    )(x)

    if bn:
        x_ = tf.keras.layers.BatchNormalization()(x_)
        x_ = tf.keras.layers.LeakyReLU(alpha=ALPHA)(x_)
    return x_


def res_block(x, channels: int) -> TTensor:
    """Residual Network Block
    In YOLOv3, the residual network block has two consecutive convolution
    layers which:
        1. half the channels with kernel size 1
        2. double the channels with kernel size 3. i.e. resume to the original
           number of channels
        3. add the original input and the output of last step

    Args:
        x (TTensor): input tensor
        channels (int): number of output (input) channels

    Returns:
        TTensor: output tensor
    """
    x_ = cnn_block(x, channels // 2, 1)
    x_ = cnn_block(x_, channels, 3)

    return x + x_


def dn53_block(x: TTensor) -> Tuple[TTensor, TTensor, TTensor]:
    """DarkNet53 block
    args:
        x (TTensor): input tensor

    Returns:
        Tuple[TTensor, TTensor, TTensor]: two intermediate results and one
            final result
            1. intermediate one: 256 out channels, after the first eight residual
               block
            2. intermediate two: 512 out channels, after the second eight
               residual blocks
            3. final result: 1024 out channels
    """
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L25-L31
    x = cnn_block(x, 32, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L33-L41
    x = cnn_block(x, 64, 3, downsample=True)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L43-L61
    x = res_block(x, 64)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L63-L71
    x = cnn_block(x, 128, 3, downsample=True)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L73-L111
    for _ in range(2):
        x = res_block(x, 128)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L113-L121
    x = cnn_block(x, 256, 3, downsample=True)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L123-L282
    for _ in range(8):
        x = res_block(x, 256)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L719-L720
    # save intermediate results
    inter_res_1 = x

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L284-L292
    x = cnn_block(x, 512, 3, downsample=True)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L294-L457
    for _ in range(8):
        x = res_block(x, 512)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L632-L633
    # save intermediate results
    inter_res_2 = x

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L459-L467
    x = cnn_block(x, 1024, 3, downsample=True)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L469-L547
    for _ in range(4):
        x = res_block(x, 1024)

    return inter_res_1, inter_res_2, x


def netv3(x: TTensor, n_class: int = 80) -> Tuple[TTensor, TTensor, TTensor]:
    """YOLOv3 network

    Args:
        x (TTensor): input tensor
        n_class (int): number of classes, default 80 (COCO)

    Returns:
        Tuple[TTensor, TTensor, TTensor]: output tensors of small, medium and
            large archors respectively
    """

    def reshape(x: TTensor) -> TTensor:
        """reshape the output as [N, W, H, 3, N_CLASS+5]"""
        shape = x.shape
        return tf.keras.layers.Reshape((shape[1], shape[2], 3, -1))(x)

    # number of channels for prediction
    pred_channels = 3 * (n_class + 5)
    # upsample with nearest neighbor interpolation method does not need to
    # learn, thereby reducing the network parameter
    upsample2 = tf.keras.layers.UpSampling2D(
        size=(2, 2),
        interpolation='nearest',
    )

    inter_1, inter_2, x_ = dn53_block(x)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L551-L557
    x_ = cnn_block(x_, 512, 1)
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L559-L565
    x_ = cnn_block(x_, 1024, 3)
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L567-L573
    x_ = cnn_block(x_, 512, 1)
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L575-L581
    x_ = cnn_block(x_, 1024, 3)
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L583-L589
    x_ = cnn_block(x_, 512, 1)
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L591-L597
    x_l = cnn_block(x_, 1024, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L599-L604
    # predict large-sized objects; shape = [None, 13, 13, 255]
    box_l = cnn_block(x_l, pred_channels, 1, bn=False)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L618-L627
    x_ = cnn_block(x_, 256, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L629-L630
    x_ = upsample2(x_)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L632-L633
    x_ = tf.concat([x_, inter_2], axis=-1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L637-L643
    x_ = cnn_block(x_, 256, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L645-L651
    x_ = cnn_block(x_, 512, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L653-L659
    x_ = cnn_block(x_, 256, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L661-L667
    x_ = cnn_block(x_, 512, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L669-L675
    x_ = cnn_block(x_, 256, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L677-L683
    x_m = cnn_block(x_, 512, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L685-L690
    # medium-sized objects, shape = [None, 26, 26, 255]
    box_m = cnn_block(x_m, pred_channels, 1, bn=False)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L705-L714
    x_ = cnn_block(x_, 128, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L716-L717
    x_ = upsample2(x_)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L719-L720
    x_ = tf.concat([x_, inter_1], axis=-1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L724-L730
    x_ = cnn_block(x_, 128, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L732-L738
    x_ = cnn_block(x_, 256, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L740-L746
    x_ = cnn_block(x_, 128, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L748-L754
    x_ = cnn_block(x_, 256, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L756-L762
    x_ = cnn_block(x_, 128, 1)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L764-L770
    x_s = cnn_block(x_, 256, 3)

    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg#L772-L777
    # predict small size objects, shape = [None, 52, 52, 255]
    box_s = cnn_block(x_s, pred_channels, 1, bn=False)

    return tuple(map(reshape, (box_s, box_m, box_l)))
