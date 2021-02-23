from base64 import standard_b64encode, standard_b64decode
from collections import namedtuple
from functools import lru_cache
import logging

import requests


MonitorQueue = namedtuple('MonitorQueue', ['name', 'limit', 'backoff_seconds'])


class QueueHTTPWrapper:

    def __init__(self, connection_params, routes=None):
        self.connection_params = connection_params
        self.message_count = {}

    def publish_messages(self, exchange, routing_key, messages, vhost="%2F"):
        for message in messages:
            self.publish_message(exchange, routing_key, message, vhost=vhost)

    def publish_message(self, exchange, routing_key, message, vhost="%2F"):
        body = {
            "properties":{},
            "routing_key": routing_key,
            "payload": standard_b64encode(message).decode('utf-8'),
            "payload_encoding": "base64"
        }
        while True:
            try:
                resp = requests.post(
                    self._pub_url(exchange, vhost),
                    json=body,
                    auth=(self.connection_params.username, self.connection_params.password),
                    timeout=3
                )
                print(resp.text)
                return
            except requests.exceptions.Timeout as err:
                logging.error('publish timeout %s', str(err))

    def get_messages(self, queue_name, vhost="%2F", count=1):
        body = {
            "count": count,
            "ackmode": "ack_requeue_false",
            "encoding":"base64",
        }
        while True:
            try:
                resp = requests.post(
                    self._get_url(queue_name, vhost),
                    json=body,
                    auth=(self.connection_params.username, self.connection_params.password),
                    timeout=3
                )
                break
            except requests.exceptions.Timeout as err:
                logging.error('consumer timeout %s', str(err))
        raw_message_list = resp.json()
        messages = []
        for raw in raw_message_list:
            self.message_count[queue_name] = raw['message_count']
            messages.append(standard_b64decode(raw.get("payload")))
        return messages

    def get_queue_size(self, queue_name):
        if queue_name in self.message_count:
            return self.message_count[queue_name]
        return 0

    def fetch_queue_size(self, queue_name, vhost="%2F"):
        resp = requests.get(self._queue_url(queue_name, vhost), auth=(self.connection_params.username, self.connection_params.password), timeout=2).json()
        self.message_count[queue_name] = resp['messages']
        return resp['messages']


    @lru_cache(maxsize=10)
    def _get_url(self, queue_name, vhost="%2F"):
        return f"http://{self.connection_params.host}:{self.connection_params.port}/api/queues/{vhost}/{queue_name}/get"

    @lru_cache(maxsize=10)
    def _queue_url(self, queue_name, vhost="%2F"):
        return f"http://{self.connection_params.host}:{self.connection_params.port}/api/queues/{vhost}/{queue_name}"

    @lru_cache(maxsize=10)
    def _pub_url(self, exchange_name, vhost="%2F"):
        return f"http://{self.connection_params.host}:{self.connection_params.port}/api/exchanges/{vhost}/{exchange_name}/publish"
