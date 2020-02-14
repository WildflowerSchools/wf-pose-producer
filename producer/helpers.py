import json
import logging
import os

from producer.settings import LOG_FORMAT, LOG_LEVEL


def get_json(outdir):
    with open(os.path.join(outdir, 'alphapose-results.json')) as fh:
        ap_json = json.load(fh)
    return ap_json


def output_json_exists(outdir):
    return os.path.exists(os.path.join(outdir, 'alphapose-results.json'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logging.basicConfig(format=LOG_FORMAT)
    return logger
