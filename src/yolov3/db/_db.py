from ._session import engine
from ._tables import boxes
from ._tables import cates
from ._tables import images

__all__ = ['DB']


class DB:
    _TABLES = (boxes, cates, images)

    @classmethod
    def create_all(cls) -> None:
        for table in reversed(cls._TABLES):
            table.create(bind=engine, checkfirst=True)

    @classmethod
    def drop_all(cls) -> None:
        """drop all tables defined in `redshift.tables`
        there's no native `DROP TABLE ... CASCADE ...` method and tables should
        be dropped from the leaves of the dependency tree back to the root
        """
        for table in cls._TABLES:
            table.drop(bind=engine, checkfirst=True)
