import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from minimal_honeycomb import MinimalHoneycombClient

from producer.helpers import get_json, output_json_exists

BATCH_SIZE = int(os.getenv("HONEYCOMB_BATCH_SIZE", 50))
ENABLE_POSEFLOW = (os.getenv("ENABLE_POSEFLOW", "yes") == "yes")

ENV = os.getenv('ENV', 'test')

GPUS = os.getenv('GPUS', '0')

HONEYCOMB_URL = os.getenv('HONEYCOMB_URL')
HONEYCOMB_TOKEN_URI = os.getenv('HONEYCOMB_TOKEN_URI')
HONEYCOMB_AUDIENCE = os.getenv('HONEYCOMB_AUDIENCE')
HONEYCOMB_CLIENT_ID = os.getenv('HONEYCOMB_CLIENT_ID')
HONEYCOMB_CLIENT_SECRET = os.getenv('HONEYCOMB_CLIENT_SECRET')

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

MODEL_NAME = "COCO-17"

# TIMEOUT = os.getenv('TIMEOUT', 3600)


class Poser(object):
    _client = None
    _pose_model_id = None

    @property
    def client(self):
        if self._client is None:
            self._client = MinimalHoneycombClient(
                uri=HONEYCOMB_URL,
                token_uri=HONEYCOMB_TOKEN_URI,
                audience=HONEYCOMB_AUDIENCE,
                client_id=HONEYCOMB_CLIENT_ID,
                client_secret=HONEYCOMB_CLIENT_SECRET
            )
        return self._client

    @property
    def pose_model_id(self):
        if self._pose_model_id is None:
            args = {
                'model_name': {
                    'type': 'String',
                    'value': MODEL_NAME
                }
            }
            response = self.client.request(
                request_type="query",
                request_name='findPoseModels',
                arguments=args,
                return_object=[
                    {'data': [
                        'pose_model_id',
                        'model_name'
                    ]}
                ]
            )
            return response['data'][0]['pose_model_id']


def parse_pose_json(pose_json, video_data):
    LOGGER.info('processing: {}'.format(str(video_data)))
    p = Poser()
    pose_model_id = p.pose_model_id
    LOGGER.info('pose_model_id: {}'.format(str(pose_model_id)))
    video_timestamp = datetime.strptime(video_data.get('timestamp'), ISO_FORMAT)

    bulk_args = {'pose2D': {'type': 'Pose2DInput', 'value': []}}
    bulk_values = []
    request_name = 'createPose2D'
    return_object = ['pose_id']

    for meta in pose_json:
        sec = int(meta.get("image_id").split('.')[0], 10)
        new_timestamp = video_timestamp + timedelta(seconds=sec * 0.1)
        keypoints = []
        joints = meta['keypoints']

        while joints:
            keypoints.append({'coordinates': [joints.pop(0), joints.pop(0)], 'quality': joints.pop(0)})
        LOGGER.info('keypoints count: {}'.format(str(len(keypoints))))

        bulk_values.append({
            'timestamp': new_timestamp.strftime(ISO_FORMAT),
            'camera': video_data.get('device_id'),
            'pose_model': pose_model_id,
            'keypoints': keypoints,
            'quality': meta.get("score"),
            'track_label': str(meta.get("idx")),
            'tags': [
                'original-timestamp: {}'.format(video_timestamp),
                'env: {}'.format(ENV)
            ]
        })

    bulk_args['pose2D']['value'] = bulk_values
    p.client.bulk_mutation(request_name, bulk_args, return_object, chunk_size=BATCH_SIZE)
    LOGGER.info("honeycomb upload complete")


def produce_poses(video_path):
    video_name = Path(video_path).resolve().stem
    LOGGER.info("video_name: {}".format(video_name))

    base_path = os.path.dirname(os.path.abspath(video_path))
    LOGGER.info("base_path: {}".format(base_path))

    outdir = os.path.join(base_path, video_name)
    LOGGER.info("outdir: {}".format(outdir))

    if output_json_exists(outdir):
        pose_json = get_json(outdir)
        if "idx" in pose_json[0] and pose_json[0]["idx"] and "idx" in pose_json[-1] and pose_json[-1]["idx"]:
            return pose_json
    cmd = [
        "python3",
        "scripts/demo_inference.py",
        "--cfg",
        "pretrained_models/256x192_res50_lr1e-3_1x.yaml",
        "--checkpoint",
        "pretrained_models/fast_res50_256x192.pth",
        "--gpus",
        str(GPUS),
        "--sp",
        "--posebatch",
        '200',
        "--detbatch",
        '10',
        "--pose_track",
        "--video",
        video_path,
        "--outdir",
        outdir
    ]
    if not ENABLE_POSEFLOW:
        cmd.remove('--pose_track')
    LOGGER.debug(cmd)
    subprocess.run(cmd)
    return get_json(outdir)


def produce_poses_job(video_data):
    pose_json = produce_poses(video_data['path'])
    # LOGGER.debug(pose_json)
    parse_pose_json(pose_json, video_data)