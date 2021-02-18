import logging
import os
import sys
import time
from collections import namedtuple

import pika
import torch.multiprocessing as mp

from producer.beta.exchanges import setup_exchanges
from producer.metric import emit


ResultTarget = namedtuple('ResultTarget', ['exchange', 'routing_key'])


class QueueWorkProcessor:

    def __init__(self, connection_params, source_queue_name, result_queue=None, batch_size=10, max_queue_size=10):
        self.connection_params = connection_params
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.source_queue_name = source_queue_name
        self.result_queue = result_queue
        mp.set_start_method('spawn')
        self.queue = mp.Queue(maxsize=self.max_queue_size)
        self._stopped = False

    def preloader(self):
        connection = pika.BlockingConnection(self.connection_params)
        channel = connection.channel()
        setup_exchanges(channel)
        while True:
            wait_time = 2
            batch = []
            while len(batch) < self.batch_size:
                if not connection.is_open:
                    logging.info("reconnecting")
                    connection = pika.BlockingConnection(self.connection_params)
                    channel = connection.channel()
                try:
                    method_frame, header_frame, body = channel.basic_get(self.source_queue_name, auto_ack=True)
                    if method_frame:
                        thing = self.prepare_single(body)
                        batch.append(thing)
                    elif len(batch) > 0:
                        break
                except pika.exceptions.ConnectionBlockedTimeout as e:
                    logging.info("timeout")
                    if len(batch) > 0:
                        break
                    time.sleep(wait_time)
                    wait_time *= wait_time
            logging.info("sending batch")
            self.queue.put(batch)

    def prepare_single(self, message):
        raise NotImplementedError("`prepare_single` has not been implemented, `QueueLoader` must be extended for purpose.")

    def processor(self):
        if self.result_queue:
            connection = pika.BlockingConnection(self.connection_params)
            channel = connection.channel()
            setup_exchanges(channel)
        while True:
            batch = self.queue.get()
            wait_time = 2
            while batch:
                try:
                    logging.info("starting batch [%s] (%s)", len(batch), self.__class__.__name__)
                    emit('QueueWorkProcessor-stats', {"batch_size": len("batch"), }, {"class":  self.__class__.__name__, "type": "batch-start"})
                    result = self.process_batch(batch)
                    logging.info("finished batch [%s] (%s)", len(result), self.__class__.__name__)
                    result = self.postprocess_batch(result)
                    emit('QueueWorkProcessor-stats', {"result_size": len("result"), }, {"class":  self.__class__.__name__, "type": "batch-complete"})
                    if len(result) > 0 and self.result_queue:
                        exchange, routing_key = self.result_queue
                        for item in result:
                            logging.info("publishing result item(%s)", (item is not None))
                            while item:
                                if not connection.is_open:
                                    logging.info("reconnecting")
                                    connection = pika.BlockingConnection(self.connection_params)
                                    channel = connection.channel()
                                try:
                                    channel.basic_publish(exchange, routing_key, item)
                                    break
                                except pika.exceptions.ConnectionBlockedTimeout as e:
                                    logging.info("timeout")
                                    time.sleep(wait_time)
                                    wait_time *= wait_time
                    del batch
                    batch = None
                except Exception as err:
                    # TODO - log the batch so it can be recovered later
                    logging.exception("failed to process [%s]", str(err))
                    del batch
                    batch = None

    def process_batch(self, batch):
        raise NotImplementedError("`process_batch` has not been implemented, `QueueLoader` must be extended for purpose.")

    def postprocess_batch(self, batch):
        return batch


    def start(self):
        return (
            self.start_worker(self.preloader),
            self.start_worker(self.processor),
        )

    def start_worker(self, target):
        p = mp.Process(target=target)
        p.start()
        return p

    @property
    def stopped(self):
        return self._stopped

    def stop(self):
        # clear queues
        self._stopped = True
        self.clear_queue()
        self._stopped = True

    def terminate(self):
        self._stopped.value = True
        self.stop()

    def clear_queue(self):
        while not self.queue.empty():
            self.queue.get()
