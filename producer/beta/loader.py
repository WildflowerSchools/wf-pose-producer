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
from producer.publisher import ReconnectingPublisher
from producer.consumer import ReconnectingConsumer

ResultTarget = namedtuple('ResultTarget', ['exchange', 'routing_key'])


class QueueWorkProcessor:

    def __init__(self, connection_params, source_queue_name, monitor_queue=None, result_queue=None, batch_size=10):
        self._connection_params = connection_params
        self.batch_size = batch_size
        self.source_queue_name = source_queue_name
        self.result_queue = result_queue
        self._consumer = None
        self._publisher = None
        self._publisher_proc = None
        self._buffer = []
        self._timer = None
        self._publish_queue = mp.Queue(maxsize=self.batch_size*2)
        self._monitor_queue = monitor_queue

    def start(self):
        if self.result_queue:
            self._publisher = ReconnectingPublisher(
                self._connection_params,
                self._publish_queue,
                self.result_queue.exchange,
                self.result_queue.routing_key,
                publish_interval=0.5,
                app_id=self.__class__.__name__,
                routes=exchanges,
                monitor_queue=self._monitor_queue,
            )
            self._publisher_proc = mp.Process(target=self._publisher)
            self._publisher_proc.start()
        self._consumer = ReconnectingConsumer(
            self._connection_params,
            self.source_queue_name,
            self,
            prefetch_count=self.batch_size,
        )
        self._consumer.run()

    def handle_message(self, message):
        if self._timer:
            self._timer.cancel()
        thing = self.prepare_single(message)
        self._buffer.append(thing)
        if len(self._buffer) >= self.batch_size:
            self._do_batch()
        else:
            self._timer = Timer(0.5, self._do_batch, args=[self])

    def _do_batch(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if len(self._buffer) > 0:
            batch = self._buffer[:self.batch_size]
            logging.info("processing batch [%i]", len(batch))
            del self._buffer[:self.batch_size]
            result = self.process_batch(batch)
            result = self.postprocess_batch(result)
            for item in result:
                self._publish_queue.put(item)
        else:
            logging.info("buffer empty")
        self._timer = Timer(0.5, self._do_batch, args=[self])

    def prepare_single(self, message):
        raise NotImplementedError("`prepare_single` has not been implemented, `QueueLoader` must be extended for purpose.")

    def process_batch(self, batch):
        raise NotImplementedError("`process_batch` has not been implemented, `QueueLoader` must be extended for purpose.")

    def postprocess_batch(self, batch):
        return batch
