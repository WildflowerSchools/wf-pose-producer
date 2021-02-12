from datetime import datetime, timedelta, date
import gc
import hashlib
import logging
import os
import re

import click

from producer import settings
from producer import helpers
from producer import tasks

LOGGER = logging.getLogger()

LOGGER.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))
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
def main():
    pass


@main.command()
@click.argument('environment_id')
@click.argument('assignment_id')
@click.argument('start')
@click.argument('duration')
@click.argument('slot')
def poses(environment_id, assignment_id, start, duration, slot, inference_model="alphapose", inference_model_version="v1"):
    from producer.tasks import execute_manifest
    LOGGER.info('processing a manifest')
    ppath = os.path.join(settings.DATA_PROCESS_DIRECTORY, environment_id, assignment_id)
    hasher = hashlib.sha1()
    hasher.update(start.encode('utf8'))
    hasher.update((datetime.strptime(start, ISO_FORMAT) + parse_duration(duration)).strftime(ISO_FORMAT).encode('utf8'))
    state_id = hasher.hexdigest()
    state_path = os.path.join(ppath, f"{state_id}.json")
    LOGGER.info(state_path)
    attempts = 0
    while attempts < settings.MAX_ATTEMPTS:
        try:
            execute_manifest(state_path, slot, inference_model, inference_model_version)
            return
        except Exception as err:
            LOGGER.exception("inference run failed: %s", err)
            attempts += 1


@main.command()
@click.argument('video_path')
def video(video_path):
    from producer.tasks import handle_video
    LOGGER.info('processing a video')
    handle_video(video_path)


@main.command('hash')
@click.argument('start')
@click.argument('duration')
def hashjob(start, duration):
    LOGGER.info('generating a hash')
    hasher = hashlib.sha1()
    hasher.update(start.encode('utf8'))
    hasher.update((datetime.strptime(start, ISO_FORMAT) + parse_duration(duration)).strftime(ISO_FORMAT).encode('utf8'))
    print(hasher.hexdigest())


@main.command()
@click.argument('environment_id')
@click.argument('assignment_id')
@click.argument('start')
@click.argument('duration')
@click.argument('slot')
@click.argument('pose_model')
@click.argument('inference_model')
@click.argument('inference_model_version')
def upload_poses(environment_id, assignment_id, start, duration, slot, pose_model="alphapose_coco18", inference_model="alphapose", inference_model_version="v1"):
    from producer.tasks import upload_manifest
    LOGGER.info('processing a manifest')
    ppath = os.path.join(settings.DATA_PROCESS_DIRECTORY, environment_id, assignment_id)
    hasher = hashlib.sha1()
    hasher.update(start.encode('utf8'))
    hasher.update((datetime.strptime(start, ISO_FORMAT) + parse_duration(duration)).strftime(ISO_FORMAT).encode('utf8'))
    state_id = hasher.hexdigest()
    state_path = os.path.join(ppath, f"{state_id}.json")
    upload_manifest(state_path, slot, pose_model, inference_model, inference_model_version)


if __name__ == '__main__':
    main()
