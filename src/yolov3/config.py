"""project config"""
import os
import sys
from logging.config import dictConfig

basedir = os.path.abspath(os.path.dirname(__file__))
srcdir = os.path.abspath(os.path.join(basedir, os.pardir))
rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))
datadir = os.path.join(rootdir, 'data')
modeldir = os.path.join(rootdir, 'model_config')


class Config:
    # pylint: disable=too-few-public-methods
    """default config"""
    # logging
    LOG_LEVEL = "INFO"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    @classmethod
    def configure_logger(cls, root_module_name):
        """configure logging"""
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

    # data
    SQLITE = os.path.join(datadir, "images.db")

    # model configs
    YOLOV3CFG = os.path.join(modeldir, "yolov3.cfg")
    YOLOV3WGT = os.path.join(modeldir, "yolov3.weights")


class TestConfig(Config):
    # pylint: disable=too-few-public-methods
    """testing config"""
    LOG_LEVEL = "DEBUG"
