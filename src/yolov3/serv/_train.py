import click

from ..datasets import Yolov3Dataset
from ..train import trainer


@click.command()
@click.option("--n-epoch", type=click.INT, required=True)
def train_yolov3(n_epoch: int):
    dataset = Yolov3Dataset()
    trainer(dataset, num_epochs=n_epoch)
