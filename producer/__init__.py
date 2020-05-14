from datetime import datetime, timedelta, date
import gc
import hashlib
import logging
import os
import re

import click

from producer.tasks import execute_manifest, upload_manifest
from producer import settings as s

LOGGER = logging.getLogger()

LOGGER.setLevel(os.getenv("LOG_LEVEL", logging.INFO))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s - %(message)s'))
LOGGER.addHandler(handler)


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
dur_regex = re.compile(r'^((?P<days>[\.\d]+?)d)?((?P<hours>[\.\d]+?)h)?((?P<minutes>[\.\d]+?)m)?((?P<seconds>[\.\d]+?)s)?$')


def parse_duration(time_str):
    parts = dur_regex.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)


@click.group()
@click.pass_context
def main(ctx):
    pass


@main.command()
@click.pass_context
@click.argument('environment_id')
@click.argument('assignment_id')
@click.argument('start')
@click.argument('duration')
@click.argument('slot')
def poses(ctx, environment_id, assignment_id, start, duration, slot):
    LOGGER.info('processing a manifest')
    ppath = os.path.join(s.DATA_PROCESS_DIRECTORY, environment_id, assignment_id)
    hasher = hashlib.sha1()
    hasher.update(start.encode('utf8'))
    hasher.update((datetime.strptime(start, ISO_FORMAT) + parse_duration(duration)).strftime(ISO_FORMAT).encode('utf8'))
    state_id = hasher.hexdigest()
    state_path = os.path.join(ppath, f"{state_id}.json")
    attempts = 0
    while attempts < s.MAX_ATTEMPTS:
        try:
            execute_manifest(state_path, slot)
            return
        except Exception as err:
            LOGGER.exception("inference run failed")
            attempts += 1


@main.command()
@click.pass_context
@click.argument('environment_id')
@click.argument('assignment_id')
@click.argument('start')
@click.argument('duration')
@click.argument('slot')
@click.argument('pose_model')
def upload_poses(ctx, environment_id, assignment_id, start, duration, slot, pose_model="alphapose_coco18"):
    LOGGER.info('processing a manifest')
    ppath = os.path.join(s.DATA_PROCESS_DIRECTORY, environment_id, assignment_id)
    hasher = hashlib.sha1()
    hasher.update(start.encode('utf8'))
    hasher.update((datetime.strptime(start, ISO_FORMAT) + parse_duration(duration)).strftime(ISO_FORMAT).encode('utf8'))
    state_id = hasher.hexdigest()
    state_path = os.path.join(ppath, f"{state_id}.json")
    upload_manifest(state_path, slot, pose_model)


if __name__ == '__main__':
    main()
