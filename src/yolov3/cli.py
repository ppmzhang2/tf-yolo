"""All commands here."""
from typing import NoReturn

import click

from .serv import coco_annot_to_csv
from .serv import create_yolo_labels
from .serv import load_coco_annot_csv
from .serv import load_darknet_yolov3
from .serv import sqlite_create_all
from .serv import sqlite_drop_all
from .serv import train_yolov3
from .serv import update_img_data


@click.group()
def cli() -> NoReturn:
    """All clicks here."""


cli.add_command(coco_annot_to_csv)
cli.add_command(create_yolo_labels)
cli.add_command(load_coco_annot_csv)
cli.add_command(load_darknet_yolov3)
cli.add_command(sqlite_create_all)
cli.add_command(sqlite_drop_all)
cli.add_command(train_yolov3)
cli.add_command(update_img_data)
