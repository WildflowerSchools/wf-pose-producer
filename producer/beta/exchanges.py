"""

video ->  detection -> box-tracker -> estimator -> pose-tracker ---> pose-upload
                                                                 |-> pose-local

"""



EXCHANGES = [
    ("videos", "video", "extract-frames"),
    ("images", "detection", "detector"),
    ("boxes", "estimator", "estimation"),
    ("boxes", "box-tracker", "catalog"),
    # ("boxes", "box-local", "catalog"),
    ("poses", "pose-tracker", "2dpose"),
    ("poses", "pose-deduplicate", "imageid"),
    ("poses", "pose-upload", "2dposeset"),
    ("poses", "pose-local", "2dposeset"),
]


def setup_exchanges(channel):
    for exchange_name, queue_name, routing_key in EXCHANGES:
        channel.exchange_declare(exchange_name)
        channel.queue_declare(queue_name)
        channel.queue_bind(
            exchange=exchange_name,
            queue=queue_name,
            routing_key=routing_key
        )
