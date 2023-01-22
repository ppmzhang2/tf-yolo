"""project config"""
import os

basedir = os.path.abspath(os.path.dirname(__file__))
srcdir = os.path.abspath(os.path.join(basedir, os.pardir))
rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))
datadir = os.path.join(rootdir, 'data')


class Config:
    # pylint: disable=too-few-public-methods
    """default config"""
    # logging
    LOG_LEVEL = "WARNING"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    # YOLO
    V3ANCHORS = (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    )
    V3ANCHORSCALES = (52, 26, 13)
    V3IN_WIDTH = 416

    # data
    SQLITE = os.path.join(datadir, "images.db")


class TestConfig(Config):
    # pylint: disable=too-few-public-methods
    """testing config"""
    LOG_LEVEL = "DEBUG"
