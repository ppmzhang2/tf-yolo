import numpy as np
import tensorflow as tf

from ..box import bbox
from ..types import Tensor
from ..types import TensorArr


def ap_1img1class(prd: TensorArr, lab: Tensor) -> Tensor:
    """AP sequence for computing mAP.

    This is for a specific image and a specific class

    TODO: output confidence

    Args:
        prd (TensorArr): prediction
        lab (TensorArr): label
    """
    if tf.size(prd) == 0 and tf.size(lab) == 0:
        return tf.constant([], dtype=tf.float32)
    if tf.size(prd) == 0:
        return tf.constant([], dtype=tf.float32)
    if tf.size(lab) == 0:
        return tf.constant([-1] * lab.shape[0], dtype=tf.float32)
    ious = np.apply_along_axis(lambda x: bbox.iou(x, lab), 1, prd)
    conf = bbox.conf(prd, squeezed=True)
    res = []
    rank_iou = np.argmax(ious, axis=-1)
    # IoU descending order indices
    indices = np.stack([np.array(range(ious.shape[0])), rank_iou], axis=1)
    rank_conf = np.argsort(-conf)  # confidence rank of predictions
    flags = {i: 0 for i in range(ious.shape[-1])}
    for i, j in indices[rank_conf]:
        if flags[j] == 0:
            res.append(ious[i, j])
            flags[j] += 1
        else:
            res.append(-1.0)
    return tf.constant(res, dtype=tf.float32)


def _unique_class(boxes: TensorArr) -> set:
    cates = bbox.class_sn(boxes, squeezed=True)
    if isinstance(cates, tf.Tensor):
        return set(np.unique(cates.numpy()))
    return set(np.unique(cates))


def nms_1class(boxes: TensorArr, iou_th: float) -> Tensor:
    """Non-maximum suppression for only one class.

    Args:
        boxes (TensorArr): 2nd order tensor
        iou_th (float): IoU threshold
    """
    confs = bbox.conf(boxes, squeezed=True)
    # indices to keep, only deleted ones should be removed
    indices_keep = tf.argsort(confs, direction="DESCENDING", axis=-1)
    # indices for tracking, both added and deleted should be removed from it
    indices_track = tf.identity(indices_keep)

    for idx1 in indices_keep:
        if tf.size(indices_track) == 0:
            break
        indices_track = indices_track[indices_track != idx1]
        for idx2 in indices_track:
            iou_ = bbox.iou(boxes[idx1], boxes[idx2])
            if iou_ > iou_th:
                indices_keep = indices_keep[indices_keep != idx2]
                indices_track = indices_track[indices_track != idx2]

    return tf.gather(boxes, indices_keep)


def nms(boxes: TensorArr, iou_th: float) -> Tensor:
    cate_sns = _unique_class(boxes)  # class SNs
    seq = []
    for sn in cate_sns:
        t = nms_1class(bbox.classof(boxes, sn), iou_th)
        if tf.size(t) != 0:
            seq.append(t)

    return tf.concat(seq, axis=0)


def _three2one_1img(
    ys: TensorArr,
    ym: TensorArr,
    yl: TensorArr,
    conf_th: float,
    iou_th: float,
    *,
    from_logits: bool = False,
) -> Tensor:
    """Combile three label tensors into one.

    The input three tensors represent three grid size of a single image.

    Args:
        ys (TensorArr): (52, 52, 3, 10)
        ym (TensorArr): (26, 26, 3, 10)
        yl (TensorArr): (13, 13, 3, 10)
        conf_th (float): confidence threshold
        iou_th (float): IoU threshold
        from_logits (bool): True if confidence score is in logit format 
    """
    if from_logits:

        def fn_objects(bx: TensorArr) -> Tensor:
            return bbox.objects(bx, from_logits=True, conf_th=conf_th)
    else:

        def fn_objects(bx: TensorArr) -> Tensor:
            return bbox.objects(bx, conf_th=conf_th)

    lab = tf.concat([fn_objects(ys), fn_objects(ym), fn_objects(yl)], axis=0)
    return nms(lab, iou_th)


def three2one(
    ys: TensorArr,
    ym: TensorArr,
    yl: TensorArr,
    conf_th: float,
    iou_th: float,
    *,
    from_logits: bool = False,
) -> tuple[Tensor, ...]:
    """Combile three label tensors into one."""
    n_batch = ys.shape[0]
    return tuple(
        _three2one_1img(
            ys[i, ...],
            ym[i, ...],
            yl[i, ...],
            conf_th,
            iou_th,
            from_logits=from_logits,
        ) for i in range(n_batch))


def ap_1class(
    prd_s: TensorArr,
    prd_m: TensorArr,
    prd_l: TensorArr,
    lab_s: TensorArr,
    lab_m: TensorArr,
    lab_l: TensorArr,
    classsn: int,
    conf_th: float,
    iou_th: float,
) -> Tensor:
    seq_prd = three2one(prd_s, prd_m, prd_l, conf_th, iou_th, from_logits=True)
    seq_lab = three2one(lab_s, lab_m, lab_l, conf_th, iou_th)
    seq_ap = []
    for prd_1img, lab_1img in zip(seq_prd, seq_lab, strict=False):
        prd_1cls1img = bbox.classof(prd_1img, classsn)
        lab_1cls1img = bbox.classof(lab_1img, classsn)
        arr = ap_1img1class(prd_1cls1img, lab_1cls1img)
        seq_ap.append(arr)
    return tf.concat(seq_ap, axis=-1)


def nbox_1class(
    lab_s: TensorArr,
    lab_m: TensorArr,
    lab_l: TensorArr,
    classsn: int,
    iou_th: float,
) -> int:
    lab_s_1class = bbox.classof(lab_s, classsn)
    lab_m_1class = bbox.classof(lab_m, classsn)
    lab_l_1class = bbox.classof(lab_l, classsn)
    lab_1class = tf.concat([lab_s_1class, lab_m_1class, lab_l_1class], axis=0)
    return nms_1class(lab_1class, iou_th).shape[0]


def unique_class(lab_s: TensorArr, lab_m: TensorArr, lab_l: TensorArr) -> set:
    return _unique_class(lab_s).union(_unique_class(lab_m)).union(
        _unique_class(lab_l))


###############################################################################
# legacy functions
###############################################################################


def pair_1class(preds: TensorArr, labels: TensorArr, classsn: int) -> Tensor:
    """TBD.

    Args:
        preds (np.ndarray): predicted boxes in shape (D_p, 10)
        labels (np.ndarray): label boxes in shape (D_l, 10)
        classsn (int): class SN

    Returns:
        Tensor: TBD
    """
    # get the specified class
    prd_class = bbox.classof(preds, classsn)
    lab_class = bbox.classof(labels, classsn)
    return ap_1img1class(prd_class, lab_class)


def pair(preds: TensorArr, labels: TensorArr) -> Tensor:
    cates_lab = _unique_class(labels)
    seq = []
    for cate in cates_lab:
        t = pair_1class(preds, labels, cate)
        seq.append(t)
    return tf.concat(seq, axis=-1)


def box_obj(
    preds: np.ndarray,
    labels: np.ndarray,
    conf_th: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Get boxes with objects.

    Returns:
        tuple[np.ndarray, np.ndarray]: prediction boxes of shape [D1, 10] and
        label boxes of shape [D2, 10]
    """
    prd_obj = bbox.objects(preds, from_logits=True, conf_th=conf_th)
    lab_obj = bbox.objects(labels, conf_th=conf_th)
    return prd_obj, lab_obj


def box_match_img(
    preds: np.ndarray,
    labels: np.ndarray,
    conf_th: float,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Get a sequence of prediction-label pairs.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], ...]: arrays are filtered with
        confidence probability threshold, all 2nd order tensors where the
        number of dimension of the 2nd rank is 10
    """
    n_batch = preds.shape[0]
    return tuple(
        box_obj(preds[i, ...], labels[i, ...], conf_th)
        for i in range(n_batch))
