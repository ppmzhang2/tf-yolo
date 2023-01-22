import click

from ..db import DB


@click.command()
def sqlite_create_all():
    DB.create_all()


@click.command()
def sqlite_drop_all():
    DB.drop_all()
