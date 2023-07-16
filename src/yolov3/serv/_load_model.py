from typing import NoReturn

import click
import cv2

from yolov3 import cfg
from yolov3.loader import load_weight_cv2
from yolov3.model import model_factory


@click.command()
@click.option("--model-out", type=click.STRING, required=True)
def load_darknet_yolov3(model_out: str) -> NoReturn:
    model = model_factory()
    net = cv2.dnn.readNetFromDarknet(cfg.YOLOV3CFG, cfg.YOLOV3WGT)
    load_weight_cv2(net, model)
    model.save(model_out)
