from glob import glob
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys
# import tracemalloc

from minimal_honeycomb import MinimalHoneycombClient

from producer import settings as s
from producer.helpers import get_json, output_json_exists
from producer.honeycomb import create_inference_execution
from producer.alpha import AlphaPoser


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


def parse_alphapose_json(num_of_keypoints, pose_json, path, device_id, timestamp, inference_execution_id):
    logging.info('processing: {}'.format(str(path)))
    if len(pose_json) == 0:
        logging.info('no poses'.format(str(path)))
        return
    p = Poser()
    pose_model_id = p.pose_model_id
    logging.info('pose_model_id: {}'.format(str(pose_model_id)))
    video_timestamp = datetime.strptime(timestamp, s.ISO_FORMAT)

    bulk_args = {'pose2D': {'type': 'Pose2DInput', 'value': []}}
    bulk_values = []
    request_name = 'createPose2D'
    return_object = ['pose_id']

    if num_of_keypoints == 18:
        for image_id in pose_json.keys():
            sec = int(image_id.split('.')[0], 10)
            new_timestamp = video_timestamp + timedelta(seconds=sec * 0.1)
            image = pose_json[image_id]
            for body in image['bodies']:
                keypoints = []
                joints = body['joints']
                score = body.get("score")
                while joints:
                    keypoints.append({'coordinates': [joints.pop(0), joints.pop(0)], 'quality': joints.pop(0)})
                payload = {
                    'timestamp': new_timestamp.strftime(s.ISO_FORMAT),
                    'camera': device_id,
                    'pose_model': pose_model_id,
                    'source': inference_execution_id,
                    'source_type': 'INFERRED',
                    'keypoints': keypoints,
                    'track_label': str(body.get("idx", "")),
                    'tags': [
                        'original-timestamp: {}'.format(video_timestamp),
                        'env: {}'.format(s.ENV)
                    ]
                }
                if score is not None:
                    payload['quality'] = score
                bulk_values.append(payload)

    if num_of_keypoints == 17:
        for body in pose_json:
            image_id = body['image_id']
            sec = int(image_id.split('.')[0], 10)
            new_timestamp = video_timestamp + timedelta(seconds=sec * 0.1)
            keypoints = []
            joints = body['keypoints']
            score = body.get("score")
            while joints:
                keypoints.append({'coordinates': [joints.pop(0), joints.pop(0)], 'quality': joints.pop(0)})
            payload = {
                'timestamp': new_timestamp.strftime(s.ISO_FORMAT),
                'camera': device_id,
                'pose_model': pose_model_id,
                'source': inference_execution_id,
                'source_type': 'INFERRED',
                'keypoints': keypoints,
                'track_label': str(body.get("idx", "")),
                'tags': [
                    'original-timestamp: {}'.format(video_timestamp),
                    'env: {}'.format(s.ENV)
                ]
            }
            if score is not None:
                payload['quality'] = score
            bulk_values.append(payload)

    bulk_args['pose2D']['value'] = bulk_values
    p.client.bulk_mutation(request_name, bulk_args, return_object, chunk_size=s.BATCH_SIZE)
    logging.info("honeycomb upload complete")


def parse_openpose_json(video_path, device_id, timestamp, inference_execution_id):
    logging.info('processing: {}'.format(str(video_path)))
    p = Poser()
    pose_model_id = p.pose_model_id
    logging.info('pose_model_id: {}'.format(str(pose_model_id)))
    video_timestamp = datetime.strptime(timestamp, s.ISO_FORMAT)

    bulk_args = {'pose2D': {'type': 'Pose2DInput', 'value': []}}
    bulk_values = []
    request_name = 'createPose2D'
    return_object = ['pose_id']

    video_name = Path(video_path).resolve().stem
    base_path = os.path.dirname(os.path.abspath(video_path))
    prefix = os.path.join(base_path, video_name)
    files = glob(f"{prefix}*.json")
    for file in files:
        sec = int(file[-20:-15])
        new_timestamp = video_timestamp + timedelta(seconds=sec * 0.1)
        with open(file, 'r') as fp:
            data = json.load(fp)
            for person in data.get("people", []):
                keypoints_raw = person.get("pose_keypoints_2d")
                keypoints = []

                while keypoints_raw:
                    keypoints.append({'coordinates': [keypoints_raw.pop(0), keypoints_raw.pop(0)], 'quality': keypoints_raw.pop(0)})

                payload = {
                    'timestamp': new_timestamp.strftime(s.ISO_FORMAT),
                    'camera': device_id,
                    'pose_model': pose_model_id,
                    'source': inference_execution_id,
                    'source_type': 'INFERRED',
                    'keypoints': keypoints,
                    'track_label': str(person.get("person_id", "")),
                    'tags': [
                        'original-timestamp: {}'.format(video_timestamp),
                        'env: {}'.format(s.ENV)
                    ]
                }
                bulk_values.append(payload)
    bulk_args['pose2D']['value'] = bulk_values
    p.client.bulk_mutation(request_name, bulk_args, return_object, chunk_size=s.BATCH_SIZE)
    logging.info("honeycomb upload complete")


def out_dir_from_path(video_path):
    video_name = Path(video_path).resolve().stem
    base_path = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(base_path, video_name)


def read_paths(path, slot):
    with open(f"{path[:-4]}{slot}.json", 'r') as reader:
        return json.load(reader)


def execute_manifest(path, slot, model, version):
    with open(path, 'r') as manipedi:
        data = json.load(manipedi)
    assignment_id = data.get("assignment_id")
    paths = read_paths(path, slot)

    if len(paths):
        inference_execution_id = data.get("inference_execution_id")
        if inference_execution_id is None:
            inference_execution_id = create_inference_execution(assignment_id, data.get("start"), data.get("end"), model=model, version=version)
            data['inference_execution_id'] = inference_execution_id
            with open(path, 'w') as fp:
                json.dump(data, fp)
                fp.flush()
        for obj in paths:
            if isinstance(obj, str):
                obj = json.loads(obj)
            video_path = obj.get("video")
            outdir = out_dir_from_path(video_path)
            if output_json_exists(outdir):
                logging.info("output exists, skipping")
            else:
                logging.info("outdir: {}".format(outdir))
                poser = AlphaPoser(
                            "pretrained_models/256x192_res50_lr1e-3_1x.yaml",
                            "pretrained_models/fast_res50_256x192.pth",
                            gpu=s.GPU,
                            single_process=True,
                            output_format="cmu",
                            pose_track=s.ALPHA_POSE_POSEFLOW
                        )
                poser.process_video(video_path, outdir)


def handle_video(video_path):
    outdir = out_dir_from_path(video_path)
    if output_json_exists(outdir):
        logging.info("output exists, skipping")
    else:
        logging.info("outdir: {}".format(outdir))
        poser = AlphaPoser(
                    "pretrained_models/256x192_res50_lr1e-3_1x.yaml",
                    "pretrained_models/fast_res50_256x192.pth",
                    gpu=s.GPU,
                    single_process=True,
                    output_format="cmu",
                    pose_track=s.ALPHA_POSE_POSEFLOW
                )
        poser.process_video(video_path, outdir)


def upload_manifest(path, slot, pose_model, model, version):
    with open(path, 'r') as manipedi:
        data = json.load(manipedi)
    assignment_id = data.get("assignment_id")
    device_id = data.get("device_id")
    paths = read_paths(path, slot)

    if len(paths):
        inference_execution_id = data.get("inference_execution_id")
        if inference_execution_id is None:
            inference_execution_id = create_inference_execution(assignment_id, data.get("start"), data.get("end"), model=model, version=version)
            data['inference_execution_id'] = inference_execution_id
            with open(path, 'w') as fp:
                json.dump(data, fp)
                fp.flush()
        if pose_model == "alphapose_coco18":
            for obj in paths:
                path = obj.get("video")
                timestamp = obj.get("timestamp")
                outdir = out_dir_from_path(path)
                pose_json = get_json(outdir)
                # TODO - push data_id into the inferenceExecution sources
                parse_alphapose_json(18, pose_json, path, device_id, timestamp, inference_execution_id)
        if pose_model == "alphapose_coco17":
            for obj in paths:
                path = obj.get("video")
                timestamp = obj.get("timestamp")
                outdir = out_dir_from_path(path)
                pose_json = get_json(outdir)
                # TODO - push data_id into the inferenceExecution sources
                parse_alphapose_json(17, pose_json, path, device_id, timestamp, inference_execution_id)
        if pose_model == "openpose_body_25":
            for obj in paths:
                path = obj.get("video")
                timestamp = obj.get("timestamp")
                parse_openpose_json(path, device_id, timestamp, inference_execution_id)
