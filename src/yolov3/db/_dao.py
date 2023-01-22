import base64
import logging
import os
from typing import Any
from typing import Optional

from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import func
from sqlalchemy.engine.row import LegacyRow
from sqlalchemy.sql import select
from sqlalchemy.sql import update

from ._session import engine
from ._tables import boxes
from ._tables import cates
from ._tables import images

LOGGER = logging.getLogger(__name__)

__all__ = ['Dao']

MAX_REC = 10000000

# format COCO bbox as a relative representation
# COCO bbox format:
#   - bbox1, bbox2: the upper-left coordinates of the bounding box
#   - bbox3, bbox3: the dimensions of your bounding box
# new format:
#   - x, y: coordinates of the bounding box center
#   - w, h: width and height of the bounding box
FORMAT_QUERY = """
SELECT box.boxid
     , box.imageid
     , box.cateid
     , box.bbox1 / img.width + 0.5 * (box.bbox3 / img.width)    AS x
     , box.bbox2 / img.height + 0.5 * (box.bbox4 / img.height)  AS y
     , box.bbox3 / img.width                                    AS w
     , box.bbox4 / img.height                                   AS h
     , cat.name                                                 AS cate_name
     , img.name                                                 AS image_name
  FROM f_box AS box
 INNER
  JOIN d_cate AS cat
    ON box.cateid = cat.cateid
 INNER
  JOIN f_image AS img
    ON box.imageid = img.imageid
"""


class Dao:

    @staticmethod
    def exec(stmt: str, *args, **kwargs):
        res = engine.execute(stmt, *args, **kwargs)
        return res

    @classmethod
    def _count(cls, column: Column) -> int:
        stmt = select([func.count(column)])
        res = cls.exec(stmt)
        return res.first()[0]

    @classmethod
    def count_box(cls) -> int:
        return cls._count(boxes.c.boxid)

    @classmethod
    def count_cate(cls) -> int:
        return cls._count(cates.c.cateid)

    @classmethod
    def count_image(cls) -> int:
        return cls._count(images.c.imageid)

    @classmethod
    def update_image(cls, imgpath: str) -> int:
        imgname = os.path.basename(imgpath)
        with open(imgpath, 'rb') as img:
            imgbytes = base64.b64encode(img.read())
        stmt = update(images).where(images.c.name == imgname).values(
            data=imgbytes)
        res = cls.exec(stmt).rowcount
        LOGGER.debug(f"{res} row(s) updated")
        return res

    @classmethod
    def update_images(cls, imgfolder: str):
        sum = 0
        LOGGER.info("updating started ...")
        for imgname in os.listdir(imgfolder):
            imgpath = os.path.join(imgfolder, imgname)
            # checking if it is a file
            if os.path.isfile(imgpath):
                cnt = cls.update_image(imgpath)
                sum += cnt
            if sum > 0 and sum % 100 == 0:
                LOGGER.info(f"    {sum} rows updated")
        LOGGER.info(f"{sum} row(s) updated")
        return sum

    @classmethod
    def _lookup(
        cls,
        table: Table,
        column: Column,
        key: Any,
    ) -> Optional[LegacyRow]:
        stmt = select([table]).where(column == key)
        res = cls.exec(stmt).first()
        return res

    @classmethod
    def lookup_image_rowid(cls, rowid: int) -> Optional[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM {images.name}
         WHERE rowid = {rowid};
        """
        return cls.exec(stmt).first()

    @classmethod
    def lookup_image_id(cls, image_id: int):
        return cls._lookup(images, images.c.imageid, image_id)

    @classmethod
    def lookup_image_name(cls, name: str):
        return cls._lookup(images, images.c.name, name)

    @classmethod
    def recreate_yolo_label(cls):
        table = 'f_yolo_label'
        stmt_drop = f"""
        DROP TABLE IF EXISTS {table};
        """
        stmt_create = f"""
        CREATE TABLE {table} AS
        {FORMAT_QUERY};
        """
        cls.exec(stmt_drop).rowcount
        return cls.exec(stmt_create).rowcount

    @classmethod
    def labels_by_img_id(cls, image_id: int) -> list[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM f_yolo_label
         WHERE imageid = {image_id};
        """
        return cls.exec(stmt).all()

    @classmethod
    def labels_by_img_name(cls, image_name: str) -> list[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM f_yolo_label
         WHERE image_name = '{image_name}';
        """
        return cls.exec(stmt).all()

    @classmethod
    def all_labels(cls, limit: int = MAX_REC) -> list[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM f_yolo_label
         LIMIT {limit};
        """
        return cls.exec(stmt).all()
