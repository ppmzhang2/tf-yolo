from typing import NoReturn

import click

from yolov3.datasets import Yolov3Dataset
from yolov3.loss import trainer
from yolov3.model import model_factory


@click.command()
@click.option("--n-epoch", type=click.INT, required=True)
def train_yolov3(n_epoch: int) -> NoReturn:
    model = model_factory()
    dataset = Yolov3Dataset()
    trainer(model, dataset, n_epoch)
