import os
import sys
from pathlib import Path
from traceback import print_exc

import click

from producer.helpers import get_logger
from producer.tasks import get_json
from producer.qlib import connect_to_rabbit, close, start_consuming


RABBIT_HOST = os.getenv("RABBIT_HOST", "localhost")
QUEUE_NAME = os.getenv("VIDEO_QUEUE_NAME", "queue-name")

logger = get_logger()


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command()
@click.pass_context
@click.option('--rabbitmq', help='hostname for rabbitmq', required=False)
@click.option('--queue', help='queue for rabbitmq', required=False)
def process_video(ctx, rabbitmq=None, queue=None):
    logger.info('running process_video')
    host = rabbitmq if rabbitmq is not None else RABBIT_HOST
    que = queue if queue is not None else QUEUE_NAME
    logger.info("attempting to connect to {}".format(host))
    channel = connect_to_rabbit(host, que)
    click.echo(dir(channel))

    try:
        logger.info("start consuming")
        start_consuming(channel, que)
    except KeyboardInterrupt:
        close(channel)
    except Exception as e:
        print_exc()
        close(channel)
        sys.exit(-1)


if __name__ == '__main__':
    main()
