# -*- coding: utf-8 -*-
# pylint: disable=C0111,C0103,R0205

from collections import namedtuple
import copy
import functools
import logging
import json

import pika


BunnyTrail = namedtuple('BunnyTrail', ['exchange', 'queue', 'routing_key'])


class AsyncConnection:
    """This is an example publisher/consumer base that will handle unexpected interactions
    with RabbitMQ such as channel and connection closures.

    If RabbitMQ closes the connection, it will reopen it. You should
    look at the output, as there are limited reasons why the connection may
    be closed, which usually are tied to permission related issues or
    socket timeouts.
    """

    def __init__(self, connection_params, routes=None):
        """Setup the example publisher object, passing in the URL we will use
        to connect to RabbitMQ.

        :param ConnectionParameters connection_params: The parameters for connecting to RabbitMQ

        """
        self.routes = routes
        self.should_reconnect = False

        self._connection = None
        self._channel = None

        self._stopping = False
        self._connection_params = connection_params
        self._routes_setup = []

    def connect(self):
        """This method connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection

        """
        logging.info('Connecting to %s', self._connection_params.host)
        return pika.SelectConnection(
            self._connection_params,
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed)

    def on_connection_open(self, _unused_connection):
        """This method is called by pika once the connection to RabbitMQ has
        been established. It passes the handle to the connection object in
        case we need it, but in this case, we'll just mark it unused.

        :param pika.SelectConnection _unused_connection: The connection

        """
        logging.info('Connection opened')
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        """This method is called by pika if the connection to RabbitMQ
        can't be established.

        :param pika.SelectConnection _unused_connection: The connection
        :param Exception err: The error

        """
        logging.error('Connection open failed: %s', err)
        self.reconnect()

    def on_connection_closed(self, _unused_connection, reason):
        """This method is invoked by pika when the connection to RabbitMQ is
        closed unexpectedly. Since it is unexpected, we will reconnect to
        RabbitMQ if it disconnects.

        :param pika.connection.Connection connection: The closed connection obj
        :param Exception reason: exception representing reason for loss of
            connection.

        """
        self._channel = None
        if self._stopping:
            self._connection.ioloop.stop()
        else:
            logging.warning('Connection closed, reconnect necessary: %s', reason)
            self.reconnect()

    def reconnect(self):
        """Will be invoked if the connection can't be opened or is
        closed. Indicates that a reconnect is necessary then stops the
        ioloop.

        """
        self.should_reconnect = True
        self.stop()

    def open_channel(self):
        """This method will open a new channel with RabbitMQ by issuing the
        Channel.Open RPC command. When RabbitMQ confirms the channel is open
        by sending the Channel.OpenOK RPC reply, the on_channel_open method
        will be invoked.

        """
        logging.info('Creating a new channel')
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        """This method is invoked by pika when the channel has been opened.
        The channel object is passed in so we can make use of it.

        Since the channel is now open, we'll declare the exchange to use.

        :param pika.channel.Channel channel: The channel object

        """
        logging.info('Channel opened')
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchanges()

    def add_on_channel_close_callback(self):
        """This method tells pika to call the on_channel_closed method if
        RabbitMQ unexpectedly closes the channel.

        """
        logging.info('Adding channel close callback')
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        """Invoked by pika when RabbitMQ unexpectedly closes the channel.
        Channels are usually closed if you attempt to do something that
        violates the protocol, such as re-declare an exchange or queue with
        different parameters. In this case, we'll close the connection
        to shutdown the object.

        :param pika.channel.Channel channel: The closed channel
        :param Exception reason: why the channel was closed

        """
        logging.warning('Channel %i was closed: %s', channel, reason)
        self._channel = None
        self.close_connection()

    def setup_exchanges(self):
        """Setup the exchange on RabbitMQ by invoking the Exchange.Declare RPC
        command. When it is complete, the on_exchange_declareok method will
        be invoked by pika.

        :param str|unicode exchange_name: The name of the exchange to declare

        """
        logging.info('Declaring exchanges')
        if self.routes:
            self._routes_setup = copy.deepcopy(self.routes)
            for route in self.routes:
                cb = functools.partial(
                    self.on_exchange_declareok, route=route)
                self._channel.exchange_declare(
                    exchange=route.exchange,
                    callback=cb)
            # do I need a safety check to maker sure on_ready is called?
        else:
            self.on_ready()

    def on_exchange_declareok(self, _unused_frame, route):
        """Invoked by pika when RabbitMQ has finished the Exchange.Declare RPC
        command.

        :param pika.Frame.Method unused_frame: Exchange.DeclareOk response frame
        :param str|unicode userdata: Extra user data (exchange name)

        """
        logging.info('Exchange declared: %s', route.exchange)
        logging.info('Declaring queue %s', route.queue)
        cb = functools.partial(
            self.on_queue_declareok, route=route)
        self._channel.queue_declare(
            queue=route.queue,
            durable=True,
            exclusive=False,
            auto_delete=False,
            callback=cb,
            arguments={"x-queue-mode": "lazy"}
        )

    def on_queue_declareok(self, _unused_frame, route):
        """Method invoked by pika when the Queue.Declare RPC call made in
        setup_queue has completed. In this method we will bind the queue
        and exchange together with the routing key by issuing the Queue.Bind
        RPC command. When this command is complete, the on_bindok method will
        be invoked by pika.

        :param pika.frame.Method method_frame: The Queue.DeclareOk frame

        """
        logging.info('Binding %s to %s with %s', route.exchange, route.queue, route.routing_key)
        cb = functools.partial(
            self.on_bindok,
            route=route
        )
        self._channel.queue_bind(
            route.queue,
            route.exchange,
            routing_key=route.routing_key,
            callback=cb
        )

    def on_bindok(self, _unused_frame, route):
        """This method is invoked by pika when it receives the Queue.BindOk
        response from RabbitMQ. Since we know we're now setup and bound, it's
        time to start publishing or consuming."""
        logging.info('Queue bound')
        self._routes_setup.remove(route)
        if self._routes_setup is not None and len(self._routes_setup) == 0:
            self.on_ready()

    def on_ready(self):
        raise NotImplementedError("`on_ready` has not been implemented.")

    def run(self):
        """Run the example code by connecting and then starting the IOLoop.

        """
        while not self._stopping:
            self._connection = None

            try:
                self._connection = self.connect()
                self._connection.ioloop.start()
            except KeyboardInterrupt:
                self.stop()
                if (self._connection is not None and
                        not self._connection.is_closed):
                    # Finish closing
                    self._connection.ioloop.start()
        logging.info('Stopped')

    def __call__(self):
        self.run()

    def stop(self):
        """Stop the example by closing the channel and connection. We
        set a flag here so that we stop scheduling new messages to be
        published. The IOLoop is started because this method is
        invoked by the Try/Catch below when KeyboardInterrupt is caught.
        Starting the IOLoop again will allow the publisher to cleanly
        disconnect from RabbitMQ.

        """
        logging.info('Stopping')
        self._stopping = True
        self.close_channel()
        self.close_connection()

    def close_channel(self):
        """Invoke this command to close the channel with RabbitMQ by sending
        the Channel.Close RPC command.

        """
        if self._channel is not None:
            logging.info('Closing the channel')
            self._channel.close()

    def close_connection(self):
        """This method closes the connection to RabbitMQ."""
        if self._connection is not None:
            logging.info('Closing connection')
            self._connection.close()
