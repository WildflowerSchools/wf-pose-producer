import json
import os


def get_json(outdir):
    with open(os.path.join(outdir, 'alphapose-results.json')) as fh:
        ap_json = json.load(fh)
    return ap_json


def output_json_exists(outdir):
    return os.path.exists(os.path.join(outdir, 'alphapose-results.json'))
