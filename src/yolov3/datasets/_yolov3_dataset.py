"""dataset for YOLOv3.

data format:
    - [N_BATCH, 416, 416, 3] for image feature matrix
    - a tuple of three arrays (for each grid scale, i.e. 52, 26, 13
      representing small, medium and large grid respectively) for labels, where
      one array has a shape like [GRID_SIZE, GRID_SIZE, N_MEASURE_PER_GRID, 6].
      The last dimension contains in order:
          x, y, w, h, conf, classid

TODO: data augmentation
"""
import base64

import cv2
import numpy as np

from .. import cfg
from .. import db
from ..box import whbox

__all__ = ["Yolov3Dataset"]

# 3*3*2 tensor repreneting anchors of 3 different scales,
# i.e. small, medium, large; and three different measures.
# ANCHORS.shape[0] represents 3 anchor scales
# ANCHORS.shape[1] represents 3 measures of each scale
#
# smalls: ANCHORS[0, ...], mediums: ANCHORS[1, ...] larges: ANCHORS[2, ...]
# [[[0.02403846, 0.03125   ],
#    [0.03846154, 0.07211538],
#    [0.07932692, 0.05528846]],
#
#   [[0.07211538, 0.14663462],
#    [0.14903846, 0.10817308],
#    [0.14182692, 0.28605769]],
#
#   [[0.27884615, 0.21634615],
#    [0.375     , 0.47596154],
#    [0.89663462, 0.78365385]]]
ANCHORS = np.array(cfg.V3_ANCHORS, dtype=np.float32) / cfg.V3_INRESOLUT

STRIDES = [int(cfg.V3_INRESOLUT // n) for n in cfg.V3_GRIDSIZE]

T_SEQ_LABEL = tuple[np.ndarray, np.ndarray, np.ndarray]

N_IMG = 40504
BATCH_COUNT_INIT = 0
IMG_ROWID_INIT = 1


class Yolov3Dataset:

    def __init__(self, batch_size: int = 2):
        self.batch_size = batch_size
        self.batch_count = BATCH_COUNT_INIT  # as a counter
        self.img_rowid = IMG_ROWID_INIT  # image row ID to stat

    def __len__(self) -> int:
        return N_IMG

    def __iter__(self):
        return self

    @staticmethod
    def imgb64_to_rgb(b64: bytes) -> np.ndarray:
        decoded_string = base64.b64decode(b64)
        nparr = np.frombuffer(decoded_string, np.uint8)
        # TODO: remove Diagnostics
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_.astype(np.float32)

    @staticmethod
    def get_label_by_id(img_id: int) -> T_SEQ_LABEL:
        """Get one label by ID.

        This method assumes that every ground truth box has a UNIQUE
        (center-cell, anchor scale, anchor measure) combination, for otherwise
        they will overwrite each other

        How to decide which scale and measure:
            criteria: IOU (only of width and height)
            by calculating the IOU between the (3, 3, 2) anchors and one box of
            shape (2, ), e.g. [0.075, 0.075], the result is (3, 3) tensor:

            [[0.13354701, 0.49309665, 0.70710654],
             [0.50122092, 0.34890323, 0.13864692],
             [0.09324138, 0.03151515, 0.00800539]]

            rank it:

            [[2, 1, 0],
             [0, 1, 2],
             [0, 1, 2]]

            the occurrence of 0s indicate the index for scale and measures:
            [0, 2], [1, 0] and [2, 0], format: [scale, measure]
        """
        # TODO: random noise?
        seq_label = [
            np.zeros((size, size, ANCHORS.shape[1], 6), dtype=np.float32)
            for size in cfg.V3_GRIDSIZE
        ]
        seq_row = db.dao.labels_by_img_id(img_id)
        for row in seq_row:
            box_wh = np.array([row.x, row.y], dtype=np.float32)
            scores = whbox.iou(box_wh, ANCHORS)
            ranks = np.argsort(-scores)  # desending
            _indices_scale, _indices_measure = np.where(ranks == 0)
            # 0: small, 1: medium, 2: large
            for idx_scale, idx_measure in zip(_indices_scale,
                                              _indices_measure,
                                              strict=True):
                stride = STRIDES[idx_scale]
                x, y = row.x * cfg.V3_INRESOLUT, row.y * cfg.V3_INRESOLUT
                # decide cell here, this is easier than drawing grids
                i, j = int(x // stride), int(y // stride)
                w, h = row.w * cfg.V3_INRESOLUT, row.h * cfg.V3_INRESOLUT
                # fill in
                seq_label[idx_scale][i, j, idx_measure, 0] = x
                seq_label[idx_scale][i, j, idx_measure, 1] = y
                seq_label[idx_scale][i, j, idx_measure, 2] = w
                seq_label[idx_scale][i, j, idx_measure, 3] = h
                seq_label[idx_scale][i, j, idx_measure, 4] = 1
                seq_label[idx_scale][i, j, idx_measure, 5] = row.cateid

        return tuple(seq_label)

    def __next__(self):
        if self.batch_count >= self.batch_size:
            self.batch_count = BATCH_COUNT_INIT
            raise StopIteration
        if self.img_rowid >= N_IMG:
            self.img_rowid = IMG_ROWID_INIT
            self.batch_count = BATCH_COUNT_INIT
            raise StopIteration

        images, labels_s, labels_m, labels_l = [], [], [], []
        while True:
            row = db.dao.lookup_image_rowid(self.img_rowid)
            if row is None:
                raise StopIteration

            rgb = self.imgb64_to_rgb(row.data)
            rgb_new = cv2.resize(
                rgb,
                (cfg.V3_INRESOLUT, cfg.V3_INRESOLUT),
                interpolation=cv2.INTER_AREA,
            )
            label_s, label_m, label_l = self.get_label_by_id(row.imageid)
            images += [rgb_new]
            labels_s += [label_s]
            labels_m += [label_m]
            labels_l += [label_l]

            # conditions
            self.batch_count += 1
            self.img_rowid += 1
            if self.batch_count >= self.batch_size:
                images_ = np.stack(images)
                labels_s_ = np.stack(labels_s)
                labels_m_ = np.stack(labels_m)
                labels_l_ = np.stack(labels_l)
                return images_, (labels_s_, labels_m_, labels_l_)
