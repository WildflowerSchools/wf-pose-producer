import click

from producer.helpers import get_logger
from producer.tasks import produce_poses
from producer.qlib import consume

LOGGER = get_logger(__name__)


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command()
@click.pass_context
@click.argument('videopath')
def generate_pose(ctx, videopath):
    LOGGER.info('running generate_pose')
    LOGGER.info('videopath: {}'.format(videopath))
    output = produce_poses(videopath)
    click.echo(output)


@main.command()
@click.pass_context
def consume_queue(ctx):
    LOGGER.info('running consume_queue')
    consume()


if __name__ == '__main__':
    main()
