import dataclasses
import json
import logging
import os
import time
from collections import namedtuple
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


PoseWorkerOptions = namedtuple("PoseWorkerOptions", ["outputpattern", "format"])

class PoseWorker(QueueWorkProcessor):

    def __init__(self, command, opts, connection_params, source_queue_name, result_queue=None, batch_size=20, max_queue_size=5):
        super().__init__(connection_params, source_queue_name, result_queue=result_queue, batch_size=batch_size, max_queue_size=max_queue_size)
        self.command = command
        self.opts = opts

    def prepare_single(self, message):
        return unpackb(message)

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
            redis_conn.set(pose_key, packb(pose))
            input_set = set(redis_conn.smembers(input_key))
            processed_set = set(redis_conn.smembers(track_key))
            if input_set == processed_set:
                result.append(packb(pose['image_id']))
        return result

    def deduplicate(self, batch):
        results = []
        redis_conn = redis.Redis(host="redis")
        for image_id in batch:
            # 1) collect the poses from redis
            input_key = f"input.{image_id}.manifest"
            box_ids = redis_conn.smembers(input_key)
            logging.info("`%s` found [[ %s ]]", input_key, box_ids)
            if len(box_ids) == 0:
                raise Exception("something is fucked")
            poses = []
            for box_id in box_ids:
                key = f"pose.{image_id}.{box_id.decode('utf8')}"
                logging.info("fetching %s from redis", key)
                poses.append(unpackb(redis_conn.get(key)))
            # 2) run pose_nms on poses
            columns = columnarize(poses, ["bbox", "score", "box_id", "keypoints", "kp_score"])
            index = index_dicts(poses, "box_id")
            logging.info("ids before filter %s", index.keys())
            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(
                    list_to_tensor(columns["bbox"]),
                    list_to_tensor(columns["score"]),
                    # pylint: disable=E1102
                    torch.tensor([UUID(box_id).bytes for box_id in columns["box_id"]], dtype=torch.uint8),
                    # pylint: enable=E1102
                    list_to_tensor(columns['keypoints']),
                    list_to_tensor(columns['kp_score']),
                    0)
            # 3) add result
            poses_clean = []
            ids = [str(UUID(bytes=bytes(box_id))) for box_id in ids]
            logging.info("ids after filter %s", ids)
            for k, box_id in enumerate(ids):
                prev = index[box_id]
                prev['bbox'] = [boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0], boxes[k][3]-boxes[k][1]]
                prev['score'] = scores[k]
                prev['keypoints'] = preds_img[k]
                prev['kp_score'] = preds_scores[k]
                poses_clean.append(prev)
            results.append(poses_clean)
        return results

    def postprocess_batch(self, batch):
        if self.command == "deduplicate":
            redis_conn = redis.Redis(host="redis")
            for poses in batch:
                if len(poses) > 0:
                    image_id = poses[0]["image_id"]
                    redis_conn.delete(f"input.{image_id}.manifest")
                    box_ids = [pose["box_id"] for pose in poses]
                    redis_conn.delete(*[f"pose.{image_id}.{box_id}" for box_id in box_ids])
                    redis_conn.delete(f"poses.{image_id}.processed")
            return [packb(poses) for poses in batch]
        return batch

    def localcache(self, batch):
        for poses in batch:
            if len(poses) > 0:
                dirname = os.path.join(os.path.dirname(poses[0]["path"]), os.path.basename(poses[0]["path"]).split('.')[0])
                os.makedirs(dirname, exist_ok=True)
                frame = PoseFrame(
                    image_id=poses[0]["image_id"],
                    image_name=poses[0]["imgname"],
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
                frame_num = frame.image_name.split('.')[0]
                # logging.info(dataclasses.asdict(frame))
                with open(os.path.join(dirname, f"poses-{frame_num}.json"), 'w') as fp:
                    json.dump(dataclasses.asdict(frame), fp)
                    fp.flush()
        return []


@click.group()
def main():
    pass


@main.command()
def rectify():
    worker = PoseWorker("rectify", None, rabbit_params(), 'pose-tracker', result_queue=ResultTarget('poses', 'imageid'))
    preloader, processor = worker.start()
    while not worker.stopped:
        time.sleep(5)


@main.command()
def deduplicate():
    worker = PoseWorker("deduplicate", None, rabbit_params(), 'pose-deduplicate', result_queue=ResultTarget('poses', '2dposeset'), batch_size=4)
    preloader, processor = worker.start()
    while not worker.stopped:
        time.sleep(5)


@main.command()
def savelocal():
    worker = PoseWorker("savelocal", None, rabbit_params(), 'pose-local', batch_size=10)
    preloader, processor = worker.start()
    while not worker.stopped:
        time.sleep(5)


if __name__ == '__main__':
    main()
