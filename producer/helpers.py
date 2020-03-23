import json
import logging
import os

from producer.settings import LOG_FORMAT, LOG_LEVEL


def get_json(outdir):
    try:
        with open(os.path.join(outdir, 'alphapose-results.json')) as fh:
            return json.load(fh)
    except:
        pass
    return None


def output_json_exists(outdir):
    return os.path.exists(os.path.join(outdir, 'alphapose-results.json'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logging.basicConfig(format=LOG_FORMAT)
    return logger
