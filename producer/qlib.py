import json

import pika
from retry import retry

from producer.tasks import produce_poses_job

from producer.helpers import get_logger

__all__ = ["connect_to_rabbit", "close"]

logger = get_logger()


@retry(pika.exceptions.AMQPConnectionError, delay=5)
def connect_to_rabbit(host, queue):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    return channel


def close(channel):
    channel.connection.close()


def job_processor(ch, method, properties, body):
    msg = json.loads(body)
    logger.info(json.dumps(msg))
    job = msg.get("job")
    logger.info('job: {}'.format(job))
    produce_poses_job(msg)
    ch.basic_ack(delivery_tag=method.delivery_tag)


def start_consuming(channel, queue_name):
    logger.info('queue_name: {}'.format(queue_name))
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=job_processor)
    channel.start_consuming()
