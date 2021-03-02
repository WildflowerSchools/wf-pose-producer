import time

import click

from producer.efque import Efque
from producer.metric import emit
from producer.beta.exchanges import exchanges


@click.command()
def main():
    client = Efque(routes=exchanges)
    while True:
        try:
            stats = client.get_stats()
            emit(f"queue-sizes", stats)
            time.sleep(1)
        except:
            # re-initialize the client
            client = Efque(routes=exchanges)


if __name__ == '__main__':
    main()
