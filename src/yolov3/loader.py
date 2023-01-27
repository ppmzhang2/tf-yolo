"""Load model weights."""
from dataclasses import dataclass
from typing import NoReturn

import cv2
import numpy as np
from keras.engine.base_layer import Layer
from keras.engine.functional import Functional

__all__ = ["load_weight_cv2"]


@dataclass(frozen=True)
class LayerMap:
    """One layer of opencv and tensorflow layer names respectively."""
    cv: str
    tf: str

    def _cv_layer(self, net: cv2.dnn.Net) -> cv2.dnn.Layer:
        return net.getLayer(self.cv)

    def _cv_weights(self, net: cv2.dnn.Net) -> tuple[np.ndarray, ...]:
        return self._cv_layer(net).blobs

    def _tf_layer(self, model: Functional) -> Layer:
        return model.get_layer(self.tf)


class BnLayerMap(LayerMap):
    """Batch Normalization Layer Mapping."""

    def load_cv2tf(self, net: cv2.dnn.Net, model: Functional) -> NoReturn:
        """Load into tensorflow model opencv BN layer weights.

        weight data format in cv: [mean, var, gamma, beta]
        weight data format in tf: [gamma, beta, mean, var]
        """
        cv_weights = self._cv_weights(net)
        mean, var, gamma, beta = (cv_weights[0], cv_weights[1], cv_weights[2],
                                  cv_weights[3])
        tf_weight = np.stack([
            gamma.squeeze(),
            beta.squeeze(),
            mean.squeeze(),
            var.squeeze(),
        ])
        tf_layer = self._tf_layer(model)
        tf_layer.set_weights(tf_weight)


class ConvLayerMap(LayerMap):
    """Convolutional Layer Mapping."""

    # length of weight sequence (blobs) if the layer enables bias
    _LEN_BIAS_LAYER_BLOB = 2

    @staticmethod
    def weight_cv2tf(arr: np.ndarray) -> np.ndarray:
        """Transform open-cv conv weights into tensorflow ones.

        permutating dimensions:
        cv Conv weigh format: (channel_out, channel_in, height, width)
        tf Conv weight format: (height, width, channel_in, channel_out)
        tf Conv bias format: (channel_in, ) i.e. squeezed vector
        """
        return arr.transpose([2, 3, 1, 0])

    def load_cv2tf(self, net: cv2.dnn.Net, model: Functional) -> NoReturn:
        """Load into tensorflow model opencv Conv layer weights.

        weight data format in tf: [weights, Optional[bias]]
        """
        cv_weights = self._cv_weights(net)
        if len(cv_weights) >= self._LEN_BIAS_LAYER_BLOB:
            weight, bias = cv_weights[0], cv_weights[1]
            tf_weights = [self.weight_cv2tf(weight), bias.squeeze()]
        else:
            weight = cv_weights[0]
            tf_weights = [self.weight_cv2tf(weight)]
        tf_layer = self._tf_layer(model)
        tf_layer.set_weights(tf_weights)


layers = [
    ConvLayerMap(cv="conv_0", tf="conv2d"),
    BnLayerMap(cv="bn_0", tf="batch_normalization"),
    ConvLayerMap(cv="conv_1", tf="conv2d_1"),
    BnLayerMap(cv="bn_1", tf="batch_normalization_1"),
    ConvLayerMap(cv="conv_2", tf="conv2d_2"),
    BnLayerMap(cv="bn_2", tf="batch_normalization_2"),
    ConvLayerMap(cv="conv_3", tf="conv2d_3"),
    BnLayerMap(cv="bn_3", tf="batch_normalization_3"),
    ConvLayerMap(cv="conv_5", tf="conv2d_4"),
    BnLayerMap(cv="bn_5", tf="batch_normalization_4"),
    ConvLayerMap(cv="conv_6", tf="conv2d_5"),
    BnLayerMap(cv="bn_6", tf="batch_normalization_5"),
    ConvLayerMap(cv="conv_7", tf="conv2d_6"),
    BnLayerMap(cv="bn_7", tf="batch_normalization_6"),
    ConvLayerMap(cv="conv_9", tf="conv2d_7"),
    BnLayerMap(cv="bn_9", tf="batch_normalization_7"),
    ConvLayerMap(cv="conv_10", tf="conv2d_8"),
    BnLayerMap(cv="bn_10", tf="batch_normalization_8"),
    ConvLayerMap(cv="conv_12", tf="conv2d_9"),
    BnLayerMap(cv="bn_12", tf="batch_normalization_9"),
    ConvLayerMap(cv="conv_13", tf="conv2d_10"),
    BnLayerMap(cv="bn_13", tf="batch_normalization_10"),
    ConvLayerMap(cv="conv_14", tf="conv2d_11"),
    BnLayerMap(cv="bn_14", tf="batch_normalization_11"),
    ConvLayerMap(cv="conv_16", tf="conv2d_12"),
    BnLayerMap(cv="bn_16", tf="batch_normalization_12"),
    ConvLayerMap(cv="conv_17", tf="conv2d_13"),
    BnLayerMap(cv="bn_17", tf="batch_normalization_13"),
    ConvLayerMap(cv="conv_19", tf="conv2d_14"),
    BnLayerMap(cv="bn_19", tf="batch_normalization_14"),
    ConvLayerMap(cv="conv_20", tf="conv2d_15"),
    BnLayerMap(cv="bn_20", tf="batch_normalization_15"),
    ConvLayerMap(cv="conv_22", tf="conv2d_16"),
    BnLayerMap(cv="bn_22", tf="batch_normalization_16"),
    ConvLayerMap(cv="conv_23", tf="conv2d_17"),
    BnLayerMap(cv="bn_23", tf="batch_normalization_17"),
    ConvLayerMap(cv="conv_25", tf="conv2d_18"),
    BnLayerMap(cv="bn_25", tf="batch_normalization_18"),
    ConvLayerMap(cv="conv_26", tf="conv2d_19"),
    BnLayerMap(cv="bn_26", tf="batch_normalization_19"),
    ConvLayerMap(cv="conv_28", tf="conv2d_20"),
    BnLayerMap(cv="bn_28", tf="batch_normalization_20"),
    ConvLayerMap(cv="conv_29", tf="conv2d_21"),
    BnLayerMap(cv="bn_29", tf="batch_normalization_21"),
    ConvLayerMap(cv="conv_31", tf="conv2d_22"),
    BnLayerMap(cv="bn_31", tf="batch_normalization_22"),
    ConvLayerMap(cv="conv_32", tf="conv2d_23"),
    BnLayerMap(cv="bn_32", tf="batch_normalization_23"),
    ConvLayerMap(cv="conv_34", tf="conv2d_24"),
    BnLayerMap(cv="bn_34", tf="batch_normalization_24"),
    ConvLayerMap(cv="conv_35", tf="conv2d_25"),
    BnLayerMap(cv="bn_35", tf="batch_normalization_25"),
    ConvLayerMap(cv="conv_37", tf="conv2d_26"),
    BnLayerMap(cv="bn_37", tf="batch_normalization_26"),
    ConvLayerMap(cv="conv_38", tf="conv2d_27"),
    BnLayerMap(cv="bn_38", tf="batch_normalization_27"),
    ConvLayerMap(cv="conv_39", tf="conv2d_28"),
    BnLayerMap(cv="bn_39", tf="batch_normalization_28"),
    ConvLayerMap(cv="conv_41", tf="conv2d_29"),
    BnLayerMap(cv="bn_41", tf="batch_normalization_29"),
    ConvLayerMap(cv="conv_42", tf="conv2d_30"),
    BnLayerMap(cv="bn_42", tf="batch_normalization_30"),
    ConvLayerMap(cv="conv_44", tf="conv2d_31"),
    BnLayerMap(cv="bn_44", tf="batch_normalization_31"),
    ConvLayerMap(cv="conv_45", tf="conv2d_32"),
    BnLayerMap(cv="bn_45", tf="batch_normalization_32"),
    ConvLayerMap(cv="conv_47", tf="conv2d_33"),
    BnLayerMap(cv="bn_47", tf="batch_normalization_33"),
    ConvLayerMap(cv="conv_48", tf="conv2d_34"),
    BnLayerMap(cv="bn_48", tf="batch_normalization_34"),
    ConvLayerMap(cv="conv_50", tf="conv2d_35"),
    BnLayerMap(cv="bn_50", tf="batch_normalization_35"),
    ConvLayerMap(cv="conv_51", tf="conv2d_36"),
    BnLayerMap(cv="bn_51", tf="batch_normalization_36"),
    ConvLayerMap(cv="conv_53", tf="conv2d_37"),
    BnLayerMap(cv="bn_53", tf="batch_normalization_37"),
    ConvLayerMap(cv="conv_54", tf="conv2d_38"),
    BnLayerMap(cv="bn_54", tf="batch_normalization_38"),
    ConvLayerMap(cv="conv_56", tf="conv2d_39"),
    BnLayerMap(cv="bn_56", tf="batch_normalization_39"),
    ConvLayerMap(cv="conv_57", tf="conv2d_40"),
    BnLayerMap(cv="bn_57", tf="batch_normalization_40"),
    ConvLayerMap(cv="conv_59", tf="conv2d_41"),
    BnLayerMap(cv="bn_59", tf="batch_normalization_41"),
    ConvLayerMap(cv="conv_60", tf="conv2d_42"),
    BnLayerMap(cv="bn_60", tf="batch_normalization_42"),
    ConvLayerMap(cv="conv_62", tf="conv2d_43"),
    BnLayerMap(cv="bn_62", tf="batch_normalization_43"),
    ConvLayerMap(cv="conv_63", tf="conv2d_44"),
    BnLayerMap(cv="bn_63", tf="batch_normalization_44"),
    ConvLayerMap(cv="conv_64", tf="conv2d_45"),
    BnLayerMap(cv="bn_64", tf="batch_normalization_45"),
    ConvLayerMap(cv="conv_66", tf="conv2d_46"),
    BnLayerMap(cv="bn_66", tf="batch_normalization_46"),
    ConvLayerMap(cv="conv_67", tf="conv2d_47"),
    BnLayerMap(cv="bn_67", tf="batch_normalization_47"),
    ConvLayerMap(cv="conv_69", tf="conv2d_48"),
    BnLayerMap(cv="bn_69", tf="batch_normalization_48"),
    ConvLayerMap(cv="conv_70", tf="conv2d_49"),
    BnLayerMap(cv="bn_70", tf="batch_normalization_49"),
    ConvLayerMap(cv="conv_72", tf="conv2d_50"),
    BnLayerMap(cv="bn_72", tf="batch_normalization_50"),
    ConvLayerMap(cv="conv_73", tf="conv2d_51"),
    BnLayerMap(cv="bn_73", tf="batch_normalization_51"),
    ConvLayerMap(cv="conv_75", tf="conv2d_52"),
    BnLayerMap(cv="bn_75", tf="batch_normalization_52"),
    ConvLayerMap(cv="conv_76", tf="conv2d_53"),
    BnLayerMap(cv="bn_76", tf="batch_normalization_53"),
    ConvLayerMap(cv="conv_77", tf="conv2d_54"),
    BnLayerMap(cv="bn_77", tf="batch_normalization_54"),
    ConvLayerMap(cv="conv_78", tf="conv2d_55"),
    BnLayerMap(cv="bn_78", tf="batch_normalization_55"),
    ConvLayerMap(cv="conv_79", tf="conv2d_56"),
    BnLayerMap(cv="bn_79", tf="batch_normalization_56"),
    ConvLayerMap(cv="conv_80", tf="conv2d_57"),
    BnLayerMap(cv="bn_80", tf="batch_normalization_57"),
    ConvLayerMap(cv="conv_81", tf="conv2d_58"),
    ConvLayerMap(cv="conv_84", tf="conv2d_59"),
    BnLayerMap(cv="bn_84", tf="batch_normalization_58"),
    ConvLayerMap(cv="conv_87", tf="conv2d_60"),
    BnLayerMap(cv="bn_87", tf="batch_normalization_59"),
    ConvLayerMap(cv="conv_88", tf="conv2d_61"),
    BnLayerMap(cv="bn_88", tf="batch_normalization_60"),
    ConvLayerMap(cv="conv_89", tf="conv2d_62"),
    BnLayerMap(cv="bn_89", tf="batch_normalization_61"),
    ConvLayerMap(cv="conv_90", tf="conv2d_63"),
    BnLayerMap(cv="bn_90", tf="batch_normalization_62"),
    ConvLayerMap(cv="conv_91", tf="conv2d_64"),
    BnLayerMap(cv="bn_91", tf="batch_normalization_63"),
    ConvLayerMap(cv="conv_92", tf="conv2d_65"),
    BnLayerMap(cv="bn_92", tf="batch_normalization_64"),
    ConvLayerMap(cv="conv_93", tf="conv2d_66"),
    ConvLayerMap(cv="conv_96", tf="conv2d_67"),
    BnLayerMap(cv="bn_96", tf="batch_normalization_65"),
    ConvLayerMap(cv="conv_99", tf="conv2d_68"),
    BnLayerMap(cv="bn_99", tf="batch_normalization_66"),
    ConvLayerMap(cv="conv_100", tf="conv2d_69"),
    BnLayerMap(cv="bn_100", tf="batch_normalization_67"),
    ConvLayerMap(cv="conv_101", tf="conv2d_70"),
    BnLayerMap(cv="bn_101", tf="batch_normalization_68"),
    ConvLayerMap(cv="conv_102", tf="conv2d_71"),
    BnLayerMap(cv="bn_102", tf="batch_normalization_69"),
    ConvLayerMap(cv="conv_103", tf="conv2d_72"),
    BnLayerMap(cv="bn_103", tf="batch_normalization_70"),
    ConvLayerMap(cv="conv_104", tf="conv2d_73"),
    BnLayerMap(cv="bn_104", tf="batch_normalization_71"),
    ConvLayerMap(cv="conv_105", tf="conv2d_74"),
]


def load_weight_cv2(net: cv2.dnn.Net, model: Functional) -> NoReturn:
    """Load weights from an open-cv model into a tensorflow one."""
    for tp in layers:
        tp.load_cv2tf(net, model)
