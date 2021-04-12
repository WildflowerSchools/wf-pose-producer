import dataclasses
import io
import json
import logging
import os
import time
from collections import namedtuple
from collections.abc import Iterable
from uuid import UUID

import click
import cv2
import numpy as np
import torch
import redis
from alphapose.utils.pPose_nms import pose_nms

from producer.beta.loader import QueueWorkProcessor, ResultTarget
from producer.beta.posemodel import PoseFrame, Pose2D, Keypoint, Box
from producer.helpers import rabbit_params, now, packb, unpackb, columnarize, index_dicts, list_to_tensor
from producer.metric import emit


PoseWorkerOptions = namedtuple("PoseWorkerOptions", ["outputpattern", "format"])


def unpackbfile(filename):
    with open(filename, 'rb') as fp:
        data = fp.read()
        return unpackb(data)


class PoseWorker(QueueWorkProcessor):

    def __init__(self, command, opts, connection_params, source_queue_name, result_queue=None, batch_size=20):
        super().__init__(connection_params, source_queue_name, result_queue=result_queue, batch_size=batch_size)
        self.command = command
        self.opts = opts

    def process_batch(self, batch):
        results = []
        if self.command == "rectify":
            results = self.rectify_poses(batch)
        if self.command == "deduplicate":
            results = self.deduplicate(batch)
        if self.command == "savelocal":
            results = self.localcache(batch)
        return results

    def rectify_poses(self, batch):
        result = []
        redis_conn = redis.Redis(host="redis")
        for pose in batch:
            input_key = f"input.{pose['image_id']}.manifest"
            pose_key = f"pose.{pose['image_id']}.{pose['box_id']}"
            track_key = f"poses.{pose['image_id']}.processed"
            redis_conn.sadd(track_key, pose['box_id'].encode('utf8'))
            obj = packb(pose)
            # bites = io.BytesIO(obj)
            name = f"pose-cache/{pose['image_id']}/{pose['box_id']}.msgpack"
            os.makedirs(f"/data/queue/queue-objects/pose-cache/{pose['image_id']}/", exist_ok=True)
            with open(f'/data/queue/queue-objects/{name}', 'wb') as fp:
                fp.write(obj)
                fp.flush()
            # self.minio_client.put_object(self.minio_bucket, name, bites, len(obj))
            input_set = set(redis_conn.smembers(input_key))
            processed_set = set(redis_conn.smembers(track_key))
            if input_set == processed_set:
                result.append(packb(pose['image_id']))
        return result

    def deduplicate(self, batch):
        results = []
        redis_conn = redis.Redis(host="redis")
        for image_id in batch:
            try:
                # 1) collect the poses from redis
                input_key = f"input.{image_id}.manifest"
                box_ids = redis_conn.smembers(input_key)
                logging.info("`%s` found [[ %s ]]", input_key, box_ids)
                poses = []
                for box_id in box_ids:
                    filename = f"/data/queue/queue-objects/pose-cache/{image_id}/{box_id.decode('utf8')}.msgpack"
                    logging.info("fetching %s from disk", filename)
                    poses.append(unpackbfile(filename))
                # 2) run pose_nms on poses
                columns = columnarize(poses, ["bbox", "score", "box_id", "keypoints", "kp_score"])
                index = index_dicts(poses, "box_id")
                logging.info("ids before filter %s", index.keys())
                boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                    pose_nms(
                        list_to_tensor(columns["bbox"]),
                        list_to_tensor(columns["score"]),
                        torch.tensor(list(range(len(columns["bbox"]))), dtype=torch.uint8),
                        list_to_tensor(columns['keypoints']),
                        list_to_tensor(columns['kp_score']),
                        0)
                # 3) add result
                poses_clean = []
                logging.info(ids)
                logging.info("ids after filter %s", ids)
                box_ids_lookup = columns["box_id"]
                logging.info(boxes)
                for k, box_index in enumerate(ids):
                    if isinstance(box_index, Iterable):
                        box_id = box_ids_lookup[box_index[0]]  # lazy, just going to grab the first one.
                    else:
                        box_id = box_ids_lookup[box_index]
                    prev = index[box_id]
                    logging.info(boxes[k])
                    prev['bbox'] = [boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0], boxes[k][3]-boxes[k][1]]
                    prev['score'] = scores[k]
                    prev['keypoints'] = preds_img[k]
                    prev['kp_score'] = preds_scores[k]
                    poses_clean.append(prev)
                results.append(poses_clean)
            except Exception as err:
                logging.exception("image_id %s deduplicate went off the rails", image_id)
                self.client.publish_message("errors", "error", packb({"error": "deduplicate_problem", "image_id": image_id, "message": str(err)}))
                continue
        return results

    def postprocess_batch(self, batch):
        if self.command == "deduplicate":
            redis_conn = redis.Redis(host="redis")
            for poses in batch:
                if len(poses) > 0:
                    image_id = poses[0]["image_id"]
                    redis_conn.unlink(f"input.{image_id}.manifest")
                    box_ids = [pose["box_id"] for pose in poses]
                    redis_conn.unlink(*[f"pose.{image_id}.{box_id}" for box_id in box_ids])
                    redis_conn.unlink(f"poses.{image_id}.processed")
            return [packb(poses) for poses in batch]
        return batch

    def localcache(self, batch):
        for poses in batch:
            if len(poses) > 0:
                dirname = os.path.join(os.path.dirname(poses[0]["path"]), os.path.basename(poses[0]["path"]).split('.')[0])
                os.makedirs(dirname, exist_ok=True)
                logging.info("output headed for %s", dirname)
                frame = PoseFrame(
                    image_id=poses[0]["image_id"],
                    image_name=poses[0]["im_name"],
                    assignment_id=poses[0]["assignment_id"],
                    environment_id=poses[0]["environment_id"],
                    timestamp=poses[0]["timestamp"],
                    video_path=poses[0]["path"],
                    poses=[],
                )
                for pose in poses:
                    joints = pose["keypoints"].tolist()
                    scores = pose["kp_score"].tolist()
                    keypoints = []
                    for index, joint in enumerate(joints):
                        keypoints.append(Keypoint(joint[0], joint[1], scores[index][0]))
                    frame.poses.append(Pose2D(
                        track_label=str(int(pose["idx"].tolist()[0])),
                        keypoints=keypoints,
                        quality=pose["proposal_score"].tolist()[0],
                        box_id=pose["box_id"],
                        bbox=Box(*pose["bbox"]),
                    ))
                emit(f"pose-stats", {
                    "poses": len(poses),
                    "frames": 1,
                })
                frame_num = frame.image_name.split('.')[0]
                file_path = os.path.join(dirname, f"poses-{frame_num}.json")
                logging.info("writring to %s", file_path)
                with open(file_path, 'w') as fp:
                    json.dump(dataclasses.asdict(frame), fp)
                    fp.flush()
        return []


@click.group()
def main():
    pass


@main.command()
def rectify():
    worker = PoseWorker("rectify", None, rabbit_params(), 'pose-tracker', result_queue=ResultTarget('poses', 'imageid'))
    worker.start()
    while True:
        time.sleep(5)


@main.command()
def deduplicate():
    worker = PoseWorker("deduplicate", None, rabbit_params(), 'pose-deduplicate', result_queue=ResultTarget('poses', '2dposeset'), batch_size=4)
    worker.start()
    while True:
        time.sleep(5)


@main.command()
def savelocal():
    worker = PoseWorker("savelocal", None, rabbit_params(), 'pose-local', batch_size=10)
    worker.start()
    while True:
        time.sleep(5)


if __name__ == '__main__':
    main()
