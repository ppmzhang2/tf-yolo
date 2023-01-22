import numpy as np

from yolov3.iou import iou_bbox

bbox1 = (2.5, 3.4, 5.0, 6.0)
bbox2 = (2.6, 4.3, 6.0, 4.0)

bboxes1 = np.array((bbox1, bbox1))
bboxes2 = np.array((bbox2, bbox2))


def test_iou_bbox():
    res = iou_bbox(bboxes1, bboxes2).numpy()
    assert np.all(res > 0.5882)
    assert np.all(res < 0.5883)
