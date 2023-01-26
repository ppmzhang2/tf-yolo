from typing import NoReturn

import click

from ..datasets import Yolov3Dataset
from ..loss import trainer
from ..model import model_factory


@click.command()
@click.option("--n-epoch", type=click.INT, required=True)
def train_yolov3(n_epoch: int) -> NoReturn:
    model = model_factory()
    dataset = Yolov3Dataset()
    trainer(model, dataset, n_epoch)
