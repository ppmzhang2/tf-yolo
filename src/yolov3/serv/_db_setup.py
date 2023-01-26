import click

from .. import db


@click.command()
def sqlite_create_all():
    db.dao.create_all()


@click.command()
def sqlite_drop_all():
    db.dao.drop_all()
