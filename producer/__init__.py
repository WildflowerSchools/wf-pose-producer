import logging
import os

import click

from producer.tasks import produce_poses
from producer.qlib import consume

RABBIT_HOST = os.getenv("RABBIT_HOST", "localhost")
QUEUE_NAME = os.getenv("VIDEO_QUEUE_NAME", "queue-name")

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)


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
