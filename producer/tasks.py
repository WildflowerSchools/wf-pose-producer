import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from minimal_honeycomb import MinimalHoneycombClient

from producer.helpers import get_json, get_logger

GPU = os.getenv('GPU', 0)

HONEYCOMB_URL = os.getenv('HONEYCOMB_URL')
HONEYCOMB_TOKEN_URI = os.getenv('HONEYCOMB_TOKEN_URI')
HONEYCOMB_AUDIENCE = os.getenv('HONEYCOMB_AUDIENCE')
HONEYCOMB_CLIENT_ID = os.getenv('HONEYCOMB_CLIENT_ID')
HONEYCOMB_CLIENT_SECRET = os.getenv('HONEYCOMB_CLIENT_SECRET')

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

MODEL_NAME = "COCO-17"

logger = get_logger()


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


def put_pose_data(args, client):
    response = client.request(
        request_type='mutation',
        request_name='createPose2D',
        arguments=args,
        return_object=['pose_id']
    )
    logger.info(str(response))


def parse_pose_json(pose_json, video_data):
    logger.info('processing: {}'.format(str(video_data)))
    p = Poser()
    pose_model_id = p.pose_model_id
    logger.info('pose_model_id: {}'.format(str(pose_model_id)))
    video_timestamp = datetime.strptime(video_data.get('timestamp'), ISO_FORMAT)

    for jpg, meta in pose_json.items():
        sec = int(jpg.split('.')[0])
        new_timestamp = video_timestamp + timedelta(seconds=sec * 0.1)
        for poses in meta['bodies']:
            logger.info('poses count: {}'.format(str(len(poses['joints']))))
            keypoints = []
            joints = poses['joints']
            while joints:
                keypoints.append({'coordinates': [joints.pop(0), joints.pop(0)], 'quality': joints.pop(0)})

            logger.info('keypoints count: {}'.format(str(len(keypoints))))
            put_pose_data({
                'pose2D': {
                    'type': 'Pose2DInput',
                    'value': {
                        'timestamp': new_timestamp.strftime(ISO_FORMAT),
                        'camera': video_data.get('device_id'),
                        'pose_model': pose_model_id,
                        'keypoints': keypoints,
                        'tags': [
                            'original-timestamp: {}'.format(video_timestamp),
                            'env: test'
                        ]
                    }
                }
            }, p.client)


def produce_poses(video_data):
    video_path = video_data['path']
    logger.info("processing video: {}".format(video_path))

    video_name = Path(video_path).resolve().stem
    logger.info("video_name: {}".format(video_name))

    base_path = os.path.dirname(os.path.abspath(video_path))
    outdir = os.path.join(base_path, video_name)

    cmd = [
        "python3",
        "scripts/demo_inference.py",
        "--cfg",
        "pretrained_models/256x192_res50_lr1e-3_1x.yaml",
        "--checkpoint",
        "pretrained_models/fast_res50_256x192.pth",
        "--gpus",
        GPU,
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
    logger.debug(cmd)

    subprocess.run(cmd, stdout=subprocess.PIPE)
    pose_json = get_json(outdir)
    parse_pose_json(pose_json, video_data)
    logger.debug(pose_json.keys())
