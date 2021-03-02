from functools import lru_cache
from threading import Timer

import redis
from tenacity import retry, wait_random, stop_after_delay


class Efque:

    def __init__(self, routes=None, redis_host="redis-queue"):
        self.redis_host = redis_host
        self._make_redis_conn()
        self.routes = routes
        self.message_count = {}

    def _make_redis_conn(self):
        self.redis_conn = redis.Redis(
            host=self.redis_host,
            retry_on_timeout=True,
            socket_timeout=10,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )

    @retry(wait=wait_random(min=1, max=4))
    def publish_messages(self, exchange, routing_key, messages):
        queues = self.get_gueues(exchange, routing_key)
        for queue_name in queues:
            self.redis_conn.rpush(queue_name, *messages)

    def publish_message(self, exchange, routing_key, message):
        self.publish_messages(exchange, routing_key, [message])

    @retry(wait=wait_random(min=1, max=4), stop=stop_after_delay(90))
    def get_messages(self, queue_name, count=1):
        return self.redis_conn.execute_command('LPOP', queue_name, count) or []

    def get_queue_size(self, queue_name):
        if queue_name in self.message_count:
            return self.message_count[queue_name]
        return self.fetch_queue_size(queue_name)

    @lru_cache(maxsize=10)
    def get_gueues(self, exchange, routing_key):
        queues = []
        for route in self.routes:
            if route.exchange == exchange and route.routing_key == routing_key:
                queues.append(route.queue)
        return queues

    @retry(wait=wait_random(min=1, max=4), stop=stop_after_delay(90))
    def fetch_queue_size(self, queue_name):
        count = self.redis_conn.llen(queue_name)
        self.message_count[queue_name] = count
        return count

    @retry(wait=wait_random(min=1, max=4), stop=stop_after_delay(90))
    def get_stats(self):
        pipe = self.redis_conn.pipeline()
        queues = list(
            {route.queue for route in self.routes}
        )
        for queue in queues:
            pipe.llen(queue)
        lengths = pipe.execute()
        return {queue: lengths[i] for i, queue in enumerate(queues)}
