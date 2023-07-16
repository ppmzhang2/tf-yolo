"""Project init."""
import os

from yolov3 import config

env = os.getenv("ENV", "test")

cfg = config.Config if env == "prod" else config.TestConfig()

cfg.configure_logger(__name__)
