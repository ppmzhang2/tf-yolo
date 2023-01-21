"""Project Configuration."""
import os
import sys
from logging.config import dictConfig
from typing import NoReturn

basedir = os.path.abspath(os.path.dirname(__file__))
srcdir = os.path.abspath(os.path.join(basedir, os.pardir))
rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))
datadir = os.path.join(rootdir, "data")
modeldir = os.path.join(rootdir, "model_config")


class Config:
    # pylint: disable=too-few-public-methods
    """Provide default Configuration."""

    # logging
    LOG_LEVEL = "INFO"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    @classmethod
    def configure_logger(cls, root_module_name: str) -> NoReturn:
        """Configure logging."""
        dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "stdout_formatter": {
                    "format": cls.LOG_LINE_FORMAT,
                    "datefmt": cls.LOG_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "stdout_handler": {
                    "level": cls.LOG_LEVEL,
                    "formatter": "stdout_formatter",
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["stdout_handler"],
                    "level": cls.LOG_LEVEL,
                    "propagate": True,
                },
            },
        })

    # YOLO constants
    # sequences all in the same order: small, medium, large
    V3_ANCHORS = (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    )
    V3_GRIDSIZE = (52, 26, 13)
    V3_INRESOLUT = 416  # input resolution: V3_INRESOLUT by V3_INRESOLUT
    V3_INCHANNELS = 3
    EPSILON = 1e-3  # avoid 0/0

    # data
    # COCO category mapping: SN, class ID, class name
    COCO_CATE = (
        (1, 1, "person"),
        (2, 2, "bicycle"),
        (3, 3, "car"),
        (4, 4, "motorcycle"),
        (5, 5, "airplane"),
        (6, 6, "bus"),
        (7, 7, "train"),
        (8, 8, "truck"),
        (9, 9, "boat"),
        (10, 10, "traffic light"),
        (11, 11, "fire hydrant"),
        (12, 13, "stop sign"),
        (13, 14, "parking meter"),
        (14, 15, "bench"),
        (15, 16, "bird"),
        (16, 17, "cat"),
        (17, 18, "dog"),
        (18, 19, "horse"),
        (19, 20, "sheep"),
        (20, 21, "cow"),
        (21, 22, "elephant"),
        (22, 23, "bear"),
        (23, 24, "zebra"),
        (24, 25, "giraffe"),
        (25, 27, "backpack"),
        (26, 28, "umbrella"),
        (27, 31, "handbag"),
        (28, 32, "tie"),
        (29, 33, "suitcase"),
        (30, 34, "frisbee"),
        (31, 35, "skis"),
        (32, 36, "snowboard"),
        (33, 37, "sports ball"),
        (34, 38, "kite"),
        (35, 39, "baseball bat"),
        (36, 40, "baseball glove"),
        (37, 41, "skateboard"),
        (38, 42, "surfboard"),
        (39, 43, "tennis racket"),
        (40, 44, "bottle"),
        (41, 46, "wine glass"),
        (42, 47, "cup"),
        (43, 48, "fork"),
        (44, 49, "knife"),
        (45, 50, "spoon"),
        (46, 51, "bowl"),
        (47, 52, "banana"),
        (48, 53, "apple"),
        (49, 54, "sandwich"),
        (50, 55, "orange"),
        (51, 56, "broccoli"),
        (52, 57, "carrot"),
        (53, 58, "hot dog"),
        (54, 59, "pizza"),
        (55, 60, "donut"),
        (56, 61, "cake"),
        (57, 62, "chair"),
        (58, 63, "couch"),
        (59, 64, "potted plant"),
        (60, 65, "bed"),
        (61, 67, "dining table"),
        (62, 70, "toilet"),
        (63, 72, "tv"),
        (64, 73, "laptop"),
        (65, 74, "mouse"),
        (66, 75, "remote"),
        (67, 76, "keyboard"),
        (68, 77, "cell phone"),
        (69, 78, "microwave"),
        (70, 79, "oven"),
        (71, 80, "toaster"),
        (72, 81, "sink"),
        (73, 82, "refrigerator"),
        (74, 84, "book"),
        (75, 85, "clock"),
        (76, 86, "vase"),
        (77, 87, "scissors"),
        (78, 88, "teddy bear"),
        (79, 89, "hair drier"),
        (80, 90, "toothbrush"),
    )
    SQLITE = os.path.join(datadir, "images.db")

    # model configs
    YOLOV3CFG = os.path.join(modeldir, "yolov3.cfg")
    YOLOV3WGT = os.path.join(modeldir, "yolov3.weights")


class TestConfig(Config):
    # pylint: disable=too-few-public-methods
    """Provide Testing Configuration."""
    LOG_LEVEL = "DEBUG"
