import json
import logging
import subprocess

import click

from .. import cfg
from ..datasets._coco_annot import CocoAnnotation
from ..db import Dao

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--file-in-json", type=click.STRING, required=True)
@click.option("--imgtag-csv", type=click.STRING, required=True)
@click.option("--cate-csv", type=click.STRING, required=True)
@click.option("--box-csv", type=click.STRING, required=True)
def coco_annot_to_csv(
    file_in_json: str,
    imgtag_csv: str,
    cate_csv: str,
    box_csv: str,
):
    with open(file_in_json, 'r') as f:
        data_json = json.load(f)
    CocoAnnotation.imgtag2csv(data_json['images'], imgtag_csv)
    CocoAnnotation.cate2csv(data_json['categories'], cate_csv)
    CocoAnnotation.box2csv(data_json['annotations'], box_csv)


@click.command()
@click.option("--imgtag-csv", type=click.STRING, required=True)
@click.option("--cate-csv", type=click.STRING, required=True)
@click.option("--box-csv", type=click.STRING, required=True)
def load_coco_annot_csv(imgtag_csv: str, cate_csv: str, box_csv: str):
    cmd_img = f".import --csv --skip 1 {imgtag_csv} f_image"
    cmd_cate = f".import --csv --skip 1 {cate_csv} d_cate"
    cmd_box = f".import --csv --skip 1 {box_csv} f_box"
    cmd = f"{cmd_img}\n{cmd_cate}\n{cmd_box}"
    sp = subprocess.Popen(
        ['sqlite3', cfg.SQLITE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = sp.communicate(input=bytes(cmd, encoding='utf8'))
    rc = sp.wait()
    LOGGER.info(f"return code = {rc} out = {out}; err = {err}")


@click.command()
@click.option("--img-folder", type=click.STRING, required=True)
def update_img_data(img_folder: str):
    Dao.update_images(img_folder)


@click.command()
def create_yolo_labels():
    Dao.recreate_yolo_label()
