import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from minimal_honeycomb import MinimalHoneycombClient

from producer import settings as s
from producer.helpers import get_json, get_logger, output_json_exists

LOGGER = get_logger(__name__)


class Poser(object):
    _client = None
    _pose_model_id = None

    @property
    def client(self):
        if self._client is None:
            self._client = MinimalHoneycombClient(
                uri=s.HONEYCOMB_URL,
                token_uri=s.HONEYCOMB_TOKEN_URI,
                audience=s.HONEYCOMB_AUDIENCE,
                client_id=s.HONEYCOMB_CLIENT_ID,
                client_secret=s.HONEYCOMB_CLIENT_SECRET
            )
        return self._client

    @property
    def pose_model_id(self):
        if self._pose_model_id is None:
            args = {
                'model_name': {
                    'type': 'String',
                    'value': s.MODEL_NAME
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
    video_timestamp = datetime.strptime(video_data.get('timestamp'), s.ISO_FORMAT)

    bulk_args = {'pose2D': {'type': 'Pose2DInput', 'value': []}}
    bulk_values = []
    request_name = 'createPose2D'
    return_object = ['pose_id']

    for image_id in pose_json.keys():
        sec = int(image_id.split('.')[0], 10)
        new_timestamp = video_timestamp + timedelta(seconds=sec * 0.1)
        image = pose_json[image_id]
        for body in image['bodies']:
            keypoints = []
            joints = body['joints']

            while joints:
                keypoints.append({'coordinates': [joints.pop(0), joints.pop(0)], 'quality': joints.pop(0)})

            bulk_values.append({
                'timestamp': new_timestamp.strftime(s.ISO_FORMAT),
                'camera': video_data.get('device_id'),
                'pose_model': pose_model_id,
                'source': video_data['inference_execution_id'],
                'source_type': 'INFERRED',
                'keypoints': keypoints,
                # 'quality': meta.get("score"),
                # 'track_label': str(meta.get("idx")),
                'tags': [
                    'original-timestamp: {}'.format(video_timestamp),
                    'env: {}'.format(s.ENV)
                ]
            })

    bulk_args['pose2D']['value'] = bulk_values
    p.client.bulk_mutation(request_name, bulk_args, return_object, chunk_size=s.BATCH_SIZE)
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
        # if s.ENABLE_POSEFLOW and "idx" in pose_json[0] and pose_json[0]["idx"] and "idx" in pose_json[-1] and pose_json[-1]["idx"]:
        if pose_json:
            return pose_json
    cmd = [
        "python3",
        "scripts/demo_inference.py",
        "--cfg",
        "pretrained_models/256x192_res50_lr1e-3_1x.yaml",
        "--checkpoint",
        "pretrained_models/fast_res50_256x192.pth",
        "--gpus",
        str(s.GPUS),
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
    if not s.ENABLE_POSEFLOW:
        cmd.remove('--pose_track')
    LOGGER.debug(cmd)
    subprocess.run(cmd)
    return get_json(outdir)


def produce_poses_job(video_data):
    LOGGER.info(f"starting job for {video_data['inference_execution_id']} inference execution")
    pose_json = produce_poses(video_data['path'])
    parse_pose_json(pose_json, video_data)

