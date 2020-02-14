import os


ENV = os.getenv('ENV', 'test')

ENABLE_POSEFLOW = (os.getenv("ENABLE_POSEFLOW", "yes") == "yes")
GPUS = os.getenv('GPUS', '0')

BATCH_SIZE = int(os.getenv("HONEYCOMB_BATCH_SIZE", 50))
HONEYCOMB_URL = os.getenv('HONEYCOMB_URL')
HONEYCOMB_TOKEN_URI = os.getenv('HONEYCOMB_TOKEN_URI')
HONEYCOMB_AUDIENCE = os.getenv('HONEYCOMB_AUDIENCE')
HONEYCOMB_CLIENT_ID = os.getenv('HONEYCOMB_CLIENT_ID')
HONEYCOMB_CLIENT_SECRET = os.getenv('HONEYCOMB_CLIENT_SECRET')

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s')
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")

MODEL_NAME = "COCO-17"

RABBIT_HOST = os.getenv("RABBIT_HOST", "localhost")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD", "guest")
RABBIT_PORT = os.getenv("RABBIT_PORT", "5672")
RABBIT_USER = os.getenv("RABBIT_USER", "guest")

EXCHANGE = os.getenv("EXCHANGE", "message")
EXCHANGE_TYPE = os.getenv("EXCHANGE_TYPE", "topic")
QUEUE = os.getenv("VIDEO_QUEUE_NAME", "queue-name")
ROUTING_KEY = os.getenv("ROUTING_KEY", QUEUE)

# TIMEOUT = os.getenv('TIMEOUT', 3600)
