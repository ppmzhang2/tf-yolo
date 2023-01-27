"""Test IoU functions of module box."""
import numpy as np

from yolov3.box import bbox

BBOX1 = (2.5, 3.4, 5.0, 6.0)
BBOX2 = (2.6, 4.3, 6.0, 4.0)
LMT_UP = 0.5883
LMT_LOW = 0.5882

bboxes1 = np.array((BBOX1, BBOX1))
bboxes2 = np.array((BBOX2, BBOX2))


def test_iou_bbox() -> None:
    """Test bbox.iou."""
    res = bbox.iou(bboxes1, bboxes2).numpy()
    assert np.all(res > LMT_LOW)
    assert np.all(res < LMT_UP)
