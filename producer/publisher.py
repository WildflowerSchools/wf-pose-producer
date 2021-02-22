# -*- coding: utf-8 -*-
# pylint: disable=C0111,C0103,R0205

from collections import namedtuple
import functools
import logging
import json
import time

import pika

from producer.pubsub import AsyncConnection


MonitorQueue = namedtuple('MonitorQueue', ['name', 'limit', 'backoff_seconds'])


class AsyncPublisher(AsyncConnection):
    """This is an example publisher that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, it will reopen it. You should
    look at the output, as there are limited reasons why the connection may
    be closed, which usually are tied to permission related issues or
    socket timeouts.

    It uses delivery confirmations and illustrates one way to keep track of
    messages that have been sent and if they've been confirmed by RabbitMQ.

    """

    def __init__(self, connection_params, queue, exchange, routing_key, routes=None, publish_interval=0.25, app_id="publisher", monitor_queue=None):
        """Setup the example publisher object, passing in the URL we will use
        to connect to RabbitMQ.

        :param str connection_params: The URL for connecting to RabbitMQ

        """
        super().__init__(connection_params, routes=routes)
        self.exchange = exchange
        self.routing_key = routing_key
        self.publish_interval = publish_interval
        self.app_id = app_id
        self.queue = queue
        self.was_publishing = False
        self._monitor_queue = monitor_queue
        self.queue_size = 0

    def on_ready(self):
        self.start_publishing()

    def start_publishing(self):
        """This method will enable delivery confirmations and schedule the
        first message to be sent to RabbitMQ

        """
        logging.info('Issuing consumer related RPC commands')
        self.get_queue_size()
        self.schedule_next_message()

    def schedule_next_message(self):
        """If we are not closing our connection to RabbitMQ, schedule another
        message to be delivered in publish_interval seconds.

        """
        logging.info('Scheduling next message for %0.1f seconds', self.publish_interval)
        self._connection.ioloop.call_later(
            self.publish_interval,
            self.publish_message
        )

    def publish_message(self):
        """If the class is not stopping, publish a message to RabbitMQ,
        appending a list of deliveries with the message number that was sent.
        This list will be used to check for delivery confirmations in the
        on_delivery_confirmations method.

        Once the message has been sent, schedule another message to be sent.
        The main reason I put scheduling in was just so you can get a good idea
        of how the process is flowing by slowing down and speeding up the
        delivery intervals by changing the publish_interval constant in the
        class.

        """
        if self._channel is None or not self._channel.is_open:
            return

        if self._monitor_queue:
            while self._monitor_queue.limit <= self.queue_size:
                logging.info("queue backed up [%s], waiting", self.queue_size)
                self._connection.ioloop.call_later(
                    self._monitor_queue.backoff_seconds,
                    self.schedule_next_message
                )
                return
        self.was_publishing = True
        properties = pika.BasicProperties(
            app_id=self.app_id,
        )
        while not self.queue.empty() and (self._monitor_queue and self._monitor_queue.limit >= self.queue_size):
            message = self.queue.get()
            self._channel.basic_publish(
                self.exchange,
                self.routing_key,
                message,
                properties
            )
            logging.info('Published message')
        self.schedule_next_message()


    def get_queue_size(self):
        if self._monitor_queue:
            self._channel.queue_declare(
                queue=self._monitor_queue.name,
                durable=True,
                exclusive=False,
                auto_delete=False,
                callback=self.on_queue_get,
                arguments={"x-queue-mode": "lazy"},
                passive=True,
            )

    def on_queue_get(self, frame):
        logging.info(frame)
        self.queue_size = frame.method.message_count
        self._connection.ioloop.call_later(
            1,
            self.get_queue_size
        )


class ReconnectingPublisher:
    """This is an example consumer that will reconnect if the nested
    AsyncConsumer indicates that a reconnect is necessary.

    """

    def __init__(self, connection_params, queue, exchange, routing_key, routes=None, publish_interval=0.25, app_id="publisher", monitor_queue=None):
        self._reconnect_delay = 0
        self._connection_params = connection_params
        self.exchange = exchange
        self.routing_key = routing_key
        self.publish_interval = publish_interval
        self.app_id = app_id
        self.queue = queue
        self._routes = routes
        self.monitor_queue = monitor_queue
        self._publisher = AsyncPublisher(
            self._connection_params,
            self.queue,
            self.exchange,
            self.routing_key,
            publish_interval=0.5,
            app_id=self.app_id,
            routes=self._routes,
            monitor_queue=self.monitor_queue
        )

    def run(self):
        while True:
            try:
                self._publisher.run()
            except KeyboardInterrupt:
                self._publisher.stop()
                break
            except pika.exceptions.ConnectionWrongStateError as err:
                self._publisher.stop()
                break
            self._maybe_reconnect()


    def __call__(self):
        self.run()

    def _maybe_reconnect(self):
        if self._publisher.should_reconnect:
            self._publisher.stop()
            reconnect_delay = self._get_reconnect_delay()
            logging.info('Reconnecting after %d seconds', reconnect_delay)
            time.sleep(reconnect_delay)
            self._publisher = AsyncPublisher(
                self._connection_params,
                self.queue,
                self.exchange,
                self.routing_key,
                publish_interval=0.5,
                app_id=self.app_id,
                routes=self._routes,
                monitor_queue=self.monitor_queue,
            )

    def _get_reconnect_delay(self):
        if self._publisher.was_publishing:
            self._reconnect_delay = 0
        else:
            self._reconnect_delay += 1
        if self._reconnect_delay > 30:
            self._reconnect_delay = 30
        return self._reconnect_delay
