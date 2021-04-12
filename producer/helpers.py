"""
Helper ulility functions used in multiple modules
"""
import copy
from datetime import datetime, date
from enum import EnumMeta
import io
from itertools import chain, islice
import json
import logging
import os

import msgpack
import numpy as np
import torch

from producer.settings import LOG_FORMAT, LOG_LEVEL


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DATE_FORMAT = "%Y-%m-%d"


def chunks(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class ObjectView:
    def __init__(self, d):
        self.__dict__ = d


def get_json(outdir):
    try:
        with open(os.path.join(outdir, 'alphapose-results.json')) as file_handle:
            return json.load(file_handle)
    except Exception as exc:
        logging.error("could not get json %s", exc)
    return None


def output_json_exists(outdir):
    return os.path.exists(os.path.join(outdir, 'alphapose-results.json'))


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logging.basicConfig(format=LOG_FORMAT)
    return logger

def now():
    return datetime.utcnow().strftime(ISO_FORMAT)


def rabbit_params():
    # return pika.ConnectionParameters(
    #     host=os.environ.get("RABBIT_HOST", "localhost"),
    #     port=int(os.environ.get("RABBIT_PORT", "5672")),
    #     credentials=pika.credentials.PlainCredentials(os.environ.get("RABBIT_USER"), os.environ.get("RABBIT_PASS")),
    #     blocked_connection_timeout=int(os.environ.get("RABBIT_BLOCK_TIMEOUT", "5")),
    #     heartbeat=int(os.environ.get("RABBIT_HEARTBEAT", "300")),
    # )
    return ObjectView({
        "host": os.environ.get("RABBIT_HOST", "localhost"),
        "port": int(os.environ.get("RABBIT_PORT", "15672")),
        "username": os.environ.get("RABBIT_USER"),
        "password": os.environ.get("RABBIT_PASS"),
    })


def decode(val):
    if val and hasattr(val, "decode"):
        return val.decode("utf8")
    return val


class CustomJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode("utf8")
        if isinstance(obj, datetime):
            return obj.strftime(ISO_FORMAT)
        if isinstance(obj, date):
            return obj.strftime(DATE_FORMAT)
        if hasattr(obj, "to_dict"):
            to_json = getattr(obj, "to_dict")
            if callable(to_json):
                return to_json()
        if hasattr(obj, "value"):
            return obj.value
        return json.JSONEncoder.default(self, obj)


def json_dumps(data, pretty=False):
    if pretty:
        return json.dumps(data, cls=CustomJsonEncoder, indent=4, sort_keys=True)
    return json.dumps(data, cls=CustomJsonEncoder)



def __encoder(obj):
    if isinstance(obj, torch.Tensor):
        bites = io.BytesIO()
        torch.save(obj, bites)
        obj = {'__torch_tensor__': bites.getvalue()}
    elif isinstance(obj, np.ndarray):
        bites = io.BytesIO()
        np.save(bites, obj)
        obj = {'__ndarray__': bites.getvalue()}
    return obj


def __decoder(obj):
    if '__torch_tensor__' in obj:
        byio = io.BytesIO(obj['__torch_tensor__'])
        obj = torch.load(byio)
    elif '__ndarray__' in obj:
        byio = io.BytesIO(obj['__ndarray__'])
        obj = np.load(byio)
    return obj


def packb(obj):
    return msgpack.packb(obj, default=__encoder, use_bin_type=True)


def unpackb(obj):
    return msgpack.unpackb(obj, object_hook=__decoder, raw=False)


def columnarize(dicts, keys):
    buffer = {k: [] for k in keys}
    for obj in dicts:
        for k in keys:
            buffer[k].append(obj.get(k))
    return buffer


def index_dicts(dicts, key, stringify=False):
    index = {}
    for obj in dicts:
        if key in obj:
            if stringify:
                index[str(obj.get(key))] = obj
            else:
                index[obj.get(key)] = obj
    return index


def list_to_tensor(seq):
    if hasattr(seq[0], 'shape'):
        tense = torch.zeros(len(seq), *seq[0].shape)
    else:
        tense = torch.zeros(len(seq))
    for index, item in enumerate(seq):
        tense[index] = item
    return tense
