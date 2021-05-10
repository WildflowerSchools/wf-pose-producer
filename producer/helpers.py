"""
Helper ulility functions used in multiple modules
"""
from datetime import datetime, date, timedelta
import io
from itertools import islice
import json
import re

import msgpack
import numpy as np
try:
    import torch
except ImportError:
    torch = None


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DATE_FORMAT = "%Y-%m-%d"



TIMEDELTA_REGEX = (r'((?P<days>-?\d+)d)?'
                   r'((?P<hours>-?\d+)h)?'
                   r'((?P<minutes>-?\d+)m)?')
TIMEDELTA_PATTERN = re.compile(TIMEDELTA_REGEX, re.IGNORECASE)


def parse_duration(time_str):
    parts = TIMEDELTA_PATTERN.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)



def chunks(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class ObjectView:
    def __init__(self, d):
        self.__dict__ = d


def now():
    return datetime.utcnow().strftime(ISO_FORMAT)


def decode(val):
    if val and hasattr(val, "decode"):
        return val.decode("utf8")
    return val


class CustomJsonEncoder(json.JSONEncoder):

    def default(self, o):
        obj = o
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
#

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
