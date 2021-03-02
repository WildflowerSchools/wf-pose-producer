import bz2
import io
import logging
import os
import sys
import time
from collections import namedtuple
import concurrent.futures
from threading import Timer
from uuid import uuid4

# from minio import Minio
# from minio.error import S3Error
import torch.multiprocessing as mp

from producer.beta.exchanges import exchanges
from producer.metric import emit
from producer.efque import Efque
from producer.helpers import unpackb, packb

ResultTarget = namedtuple('ResultTarget', ['exchange', 'routing_key'])


class QueueWorkProcessor:

    def __init__(self, connection_params, source_queue_name, monitor_queue=None, result_queue=None, batch_size=10):
        self._connection_params = connection_params
        self.batch_size = batch_size
        self.source_queue_name = source_queue_name
        self.result_queue = result_queue
        self.client = Efque(routes=exchanges)
        self._monitor_queue = monitor_queue
        # self.minio_client = Minio(
        #     "minio:9000",
        #     access_key=os.environ.get("MINIO_ACCESS_KEY"),
        #     secret_key=os.environ.get("MINIO_SECRET_KEY"),
        #     secure=False,
        # )
        self.minio_bucket = "queue-objects"
        for route in exchanges:
            os.makedirs(f'/data/queue/queue-objects/{route.exchange}/{route.routing_key}/', exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.futures = []

    def start(self):
        while True:
            logging.info(self.futures)
            self.futures = [future for future in self.futures if not future.done()]
            while len(self.futures) > self.batch_size * 2:
                logging.info("too many futures (%i) waiting", len(self.futures))
                time.sleep(0.01)
                self.futures = [future for future in self.futures if not future.done()]
                logging.info(self.futures)

            if self._monitor_queue:
                logging.info("checking %s length, should be less than %i", self._monitor_queue.name, self._monitor_queue.limit)
                size = self.client.fetch_queue_size(self._monitor_queue.name)
                while size >= self._monitor_queue.limit:
                    logging.info("... queue backed up [%s], waiting", size)
                    time.sleep(self._monitor_queue.backoff_seconds)
                    size = self.client.fetch_queue_size(self._monitor_queue.name)
            batch = self.client.get_messages(self.source_queue_name, count=self.batch_size)
            if len(batch) == 0:
                logging.info("empty queue")
                time.sleep(1)
            else:
                logging.info("processing batch [%i]", len(batch))
                batch = [self.prepare_single(item) for item in batch]
                result = self.process_batch(batch)
                logging.info("processed [%i]", len(result))
                if result and len(result) > 0:
                    logging.info("handling results")
                    result = self.postprocess_batch(result)
                    if result and len(result) > 0 and self.result_queue:
                        self.make_refs(result)

    def make_refs(self, objs):
        result_queue = self.result_queue
        client = self.client
        logging.info("making refs")

        def save_obj(obj):
            name = f"{result_queue.exchange}/{result_queue.routing_key}/{str(uuid4())}"
            with open(f'/data/queue/queue-objects/{name}', 'wb') as fp:
                fp.write(bz2.compress(obj))
                fp.flush()
            logging.info("saving %s", name)
            client.publish_message(
                result_queue.exchange,
                result_queue.routing_key,
                packb({"ref": name})
            )

        for obj in objs:
            self.futures.append(
                self.executor.submit(save_obj, obj)
            )

    def prepare_single(self, message):
        mess = unpackb(message)
        if "ref" in mess:
            with open(f'/data/queue/queue-objects/{mess.get("ref")}', 'rb') as fp:
                data = bz2.decompress(fp.read())
                mess = unpackb(data)
        return mess

    def process_batch(self, batch):
        raise NotImplementedError("`process_batch` has not been implemented, `QueueLoader` must be extended for purpose.")

    def postprocess_batch(self, batch):
        return batch
