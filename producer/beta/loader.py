import logging
import os
import sys
import time
from collections import namedtuple
from threading import Timer

import pika
import torch.multiprocessing as mp

from producer.beta.exchanges import exchanges
from producer.metric import emit
# from producer.publisher import ReconnectingPublisher
# from producer.consumer import ReconnectingConsumer
from producer.hyperbunny import QueueHTTPWrapper as BunnyLove


ResultTarget = namedtuple('ResultTarget', ['exchange', 'routing_key'])


class QueueWorkProcessor:

    def __init__(self, connection_params, source_queue_name, monitor_queue=None, result_queue=None, batch_size=10):
        self._connection_params = connection_params
        self.batch_size = batch_size
        self.source_queue_name = source_queue_name
        self.result_queue = result_queue
        self.client = BunnyLove(self._connection_params, exchanges)
        self._monitor_queue = monitor_queue

    def start(self):
        while True:
            batch = self.client.get_messages(self.source_queue_name, count=self.batch_size)
            if len(batch) == 0:
                logging.info("empty queue")
                time.sleep(1)
            else:
                logging.info("processing batch [%i]", len(batch))
                batch = [self.prepare_single(item) for item in batch]
                result = self.process_batch(batch)
                result = self.postprocess_batch(result)
                if self.result_queue:
                    self.client.publish_messages(
                        self.result_queue.exchange,
                        self.result_queue.routing_key,
                        result
                    )
                if self._monitor_queue:
                    size = self.client.get_queue_size(self._monitor_queue.name)
                    while self._monitor_queue.limit <= size:
                        logging.info("... queue backed up [%s], waiting", size)
                        time.sleep(self._monitor_queue.backoff_seconds)
                        size = self.client.fetch_queue_size(self._monitor_queue.name)

    def prepare_single(self, message):
        raise NotImplementedError("`prepare_single` has not been implemented, `QueueLoader` must be extended for purpose.")

    def process_batch(self, batch):
        raise NotImplementedError("`process_batch` has not been implemented, `QueueLoader` must be extended for purpose.")

    def postprocess_batch(self, batch):
        return batch
