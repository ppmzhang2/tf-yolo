import base64
import logging
import os

from sqlalchemy import Column
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy import func
from sqlalchemy.engine.cursor import LegacyCursorResult
from sqlalchemy.engine.row import LegacyRow
from sqlalchemy.sql import select
from sqlalchemy.sql import update

from .. import cfg
from ._tables import boxes
from ._tables import cates
from ._tables import images

LOGGER = logging.getLogger(__name__)

__all__ = ["dao"]

MAX_REC = 10000000

# format COCO box as a relative representation
# COCO box format:
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

_TABLES = (boxes, cates, images)


class SingletonMeta(type):
    """Singleton meta-class."""
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Dao(metaclass=SingletonMeta):

    __slots__ = ["_engine"]

    def __init__(self):
        self._engine = create_engine(f"sqlite:///{cfg.SQLITE}")

    def create_all(self) -> None:
        for table in reversed(_TABLES):
            table.create(bind=self._engine, checkfirst=True)

    def drop_all(self) -> None:
        """Drop all tables defined in `redshift.tables`.

        there's no native `DROP TABLE ... CASCADE ...` method and tables should
        be dropped from the leaves of the dependency tree back to the root
        """
        for table in _TABLES:
            table.drop(bind=self._engine, checkfirst=True)

    def _exec(self, stmt: str, *args, **kwargs) -> LegacyCursorResult:
        return self._engine.execute(stmt, *args, **kwargs)

    def _count(self, column: Column) -> int:
        stmt = select([func.count(column)])
        res = self._exec(stmt)
        return res.first()[0]

    def count_box(self) -> int:
        return self._count(boxes.c.boxid)

    def count_cate(self) -> int:
        return self._count(cates.c.cateid)

    def count_image(self) -> int:
        return self._count(images.c.imageid)

    def update_image(self, imgpath: str) -> int:
        imgname = os.path.basename(imgpath)
        with open(imgpath, "rb") as img:
            imgbytes = base64.b64encode(img.read())
        stmt = update(images).where(images.c.name == imgname).values(
            data=imgbytes)
        res = self._exec(stmt).rowcount
        LOGGER.debug(f"{res} row(s) updated")
        return res

    def update_images(self, imgfolder: str) -> int:
        row_total = 0
        LOGGER.info("updating started ...")
        for imgname in os.listdir(imgfolder):
            imgpath = os.path.join(imgfolder, imgname)
            # checking if it is a file
            if os.path.isfile(imgpath):
                cnt = self.update_image(imgpath)
                row_total += cnt
            if row_total > 0 and row_total % 100 == 0:
                LOGGER.info(f"    {row_total} rows updated")
        LOGGER.info(f"{row_total} row(s) updated")
        return row_total

    def _lookup(
        self,
        table: Table,
        column: Column,
        key: str | int | float,
    ) -> LegacyRow | None:
        stmt = select([table]).where(column == key)
        return self._exec(stmt).first()

    def lookup_image_rowid(self, rowid: int) -> LegacyRow | None:
        stmt = f"""
        SELECT *
          FROM {images.name}
         WHERE rowid = {rowid};
        """
        return self._exec(stmt).first()

    def lookup_image_id(self, image_id: int) -> LegacyRow | None:
        return self._lookup(images, images.c.imageid, image_id)

    def lookup_image_name(self, name: str) -> LegacyRow | None:
        return self._lookup(images, images.c.name, name)

    def recreate_yolo_label(self) -> int:
        table = "f_yolo_label"
        stmt_drop = f"""
        DROP TABLE IF EXISTS {table};
        """
        stmt_create = f"""
        CREATE TABLE {table} AS
        {FORMAT_QUERY};
        """
        self._exec(stmt_drop).rowcount
        return self._exec(stmt_create).rowcount

    def labels_by_img_id(self, image_id: int) -> list[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM f_yolo_label
         WHERE imageid = {image_id};
        """
        return self._exec(stmt).all()

    def labels_by_img_name(self, image_name: str) -> list[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM f_yolo_label
         WHERE image_name = '{image_name}';
        """
        return self._exec(stmt).all()

    def all_labels(self, limit: int = MAX_REC) -> list[LegacyRow]:
        stmt = f"""
        SELECT *
          FROM f_yolo_label
         LIMIT {limit};
        """
        return self._exec(stmt).all()

    def categories(self) -> dict[int, str]:
        stmt = f"""
        SELECT *
          FROM {cates};
        """
        rows = self._exec(stmt).all()
        return {r[cates.c.cateid.name]: r[cates.c.name.name] for r in rows}


dao = Dao()
