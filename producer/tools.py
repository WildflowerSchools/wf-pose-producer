import logging

import click
import msgpack
import pika

from producer.helpers import rabbit_params, now, packb, unpackb
from producer.beta.exchanges import setup_exchanges


@click.group()
def main():
    pass


@main.command()
@click.option('--path')
@click.option('--assignment_id')
@click.option('--environment_id')
@click.option('--timestamp')
def queue_video(path, assignment_id, environment_id, timestamp):
    logging.info("queueing video")
    logging.info(path)
    connection = pika.BlockingConnection(rabbit_params())
    channel = connection.channel()
    setup_exchanges(channel)
    body = packb({
        "date": now(),
        "path": path,
        "assignment_id": assignment_id,
        "environment_id": environment_id,
        "timestamp": timestamp,
    })
    channel.basic_publish("videos", "extract-frames", body)
    logging.info("done")


@main.command()
@click.option('--queue')
def read_queue_message(queue):
    logging.info("reading from queue %s", queue)
    connection = pika.BlockingConnection(rabbit_params())
    channel = connection.channel()
    setup_exchanges(channel)
    method_frame, header_frame, body = channel.basic_get(queue, auto_ack=False)
    data = unpackb(body)
    print(data)


if __name__ == '__main__':
    main()
