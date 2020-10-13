import os


ENV = os.getenv('ENV', 'test')

GPU = os.getenv('GPU', None)

BATCH_SIZE = int(os.getenv("HONEYCOMB_BATCH_SIZE", 50))
HONEYCOMB_URL = os.getenv('HONEYCOMB_URL')
HONEYCOMB_TOKEN_URI = os.getenv('HONEYCOMB_TOKEN_URI')
HONEYCOMB_AUDIENCE = os.getenv('HONEYCOMB_AUDIENCE')
HONEYCOMB_CLIENT_ID = os.getenv('HONEYCOMB_CLIENT_ID')
HONEYCOMB_CLIENT_SECRET = os.getenv('HONEYCOMB_CLIENT_SECRET')

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) -35s %(lineno) -5d: %(message)s')
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")

MODEL_NAME = "COCO-18"

RABBIT_HOST = os.getenv("RABBIT_HOST", "localhost")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD", "guest")
RABBIT_PORT = os.getenv("RABBIT_PORT", "5672")
RABBIT_USER = os.getenv("RABBIT_USER", "guest")

EXCHANGE = os.getenv("EXCHANGE", "message")
EXCHANGE_TYPE = os.getenv("EXCHANGE_TYPE", "topic")
QUEUE = os.getenv("VIDEO_QUEUE_NAME", "queue-name")
ROUTING_KEY = os.getenv("ROUTING_KEY", QUEUE)
PREFETCH_COUNT = int(os.getenv("PREFETCH_COUNT", 1))

# TIMEOUT = os.getenv('TIMEOUT', 3600)

ALPHA_POSE_POSEFLOW = os.getenv("ALPHA_POSE_POSEFLOW", "false") == "true"
ALPHA_POSE_POSEFLOW = os.getenv("ALPHA_POSE_CHECKPOINT", "pretrained_models/256x192_res50_lr1e-3_1x.yaml")
ALPHA_POSE_POSEFLOW = os.getenv("ALPHA_POSE_CONFIG", "pretrained_models/fast_res50_256x192.pth")

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 4))

DATA_PROCESS_DIRECTORY = os.getenv("DATA_PROCESS_DIRECTORY", "/data")
