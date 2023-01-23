import click

from ..dao import dao


@click.command()
def sqlite_create_all():
    dao.create_all()


@click.command()
def sqlite_drop_all():
    dao.drop_all()
