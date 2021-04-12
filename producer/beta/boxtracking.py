from collections.abc import Iterable
import dataclasses
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import torch

from alphapose.utils.pPose_nms import pose_nms
from producer.helpers import packb, unpackb, list_to_tensor, columnarize, index_dicts
from producer.metric import emit
from producer.beta.posemodel import PoseFrame, Pose2D, Keypoint, Box


class BoxTracker:

    def __init__(self):
        self.image_index = sqlite3.connect(":memory:")
        cur = self.image_index.cursor()
        cur.execute('CREATE TABLE image_boxes (img_id text, box_id text)')
        cur.execute('CREATE TABLE pose_boxes (img_id text, box_id text, pose_data blob)')
        cur.execute('CREATE TABLE images (img_id text, meta blob)')
        self.image_index.commit()

    def declare_images(self, images):
        cur = self.image_index.cursor()
        for image in images:
            obj = packb({
                "image_id": image['image_id'],
                "im_name": image['im_name'],
                "assignment_id": image['assignment_id'],
                "environment_id": image['environment_id'],
                "timestamp": image['timestamp'],
                "path": image['path'],
            })
            cur.execute('INSERT INTO images VALUES (?, ?)', (image['image_id'], obj))
        self.image_index.commit()

    def ingest_boxes(self, batch):
        cur = self.image_index.cursor()
        results = []
        for image in batch:
            base = {
                "image_id": image['image_id'],
                "orig_img": image["orig_img"],
                "im_name": image["im_name"],
                "path": image["path"],
                "date": image["date"],
                "assignment_id": image["assignment_id"],
                "environment_id": image["environment_id"],
                "timestamp": image["timestamp"],
            }
            box_ids = []
            for box in image["boxes"]:
                box.update(base)
                box_id = str(uuid4())
                box["box_id"] = box_id
                box_ids.append((image['image_id'], box_id, ))
                results.append(box)
            cur.executemany('INSERT INTO image_boxes VALUES (?,?)', box_ids)
        self.image_index.commit()
        return results

    def rectify_poses(self, batch):
        cur = self.image_index.cursor()
        for pose in batch:
            obj = packb(pose)
            cur.execute('INSERT INTO pose_boxes VALUES (?, ?, ?)', (pose['image_id'], pose['box_id'], obj))
        self.image_index.commit()

    def get_image_ids(self):
        cur = self.image_index.cursor()
        ibx = cur.execute('SELECT img_id FROM images').fetchall()
        result = {ii[0] for ii in ibx}
        return result

    def get_poses_for_image_id(self, image_id):
        cur = self.image_index.cursor()
        ibx = cur.execute('SELECT pose_data FROM pose_boxes WHERE img_id = ?', (image_id, )).fetchall()
        result = [unpackb(ii[0]) for ii in ibx]
        return result

    def get_image_meta(self, img_id):
        cur = self.image_index.cursor()
        result = cur.execute('SELECT meta FROM images WHERE img_id = ?', (img_id, )).fetchone()
        return unpackb(result[0])


def check_for_output(path):
    dirname = os.path.join(os.path.dirname(path), os.path.basename(path).split('.')[0])
    if os.path.exists(dirname):
        p = Path(dirname)
        return {f.stem for f in p.iterdir()}
    return set()


def drop_tombstone(image):
    dirname = os.path.join(os.path.dirname(image["path"]), os.path.basename(image["path"]).split('.')[0])
    os.makedirs(dirname, exist_ok=True)
    logging.info("output headed for %s", dirname)
    frame = PoseFrame(
        image_id=image["image_id"],
        image_name=image["im_name"],
        assignment_id=image["assignment_id"],
        environment_id=image["environment_id"],
        timestamp=image["timestamp"],
        video_path=image["path"],
        poses=[],
    )
    frame_num = frame.image_name.split('.')[0]
    file_path = os.path.join(dirname, f"poses-{frame_num}.json")
    logging.info("writring to %s", file_path)
    with open(file_path, 'w') as fp:
        json.dump(dataclasses.asdict(frame), fp)
        fp.flush()


def run_nms(poses):
    try:
        # prepare output location
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

        columns = columnarize(poses, ["bbox", "score", "box_id", "keypoints", "kp_score"])
        index = index_dicts(poses, "box_id")
        logging.debug("ids before filter %s", index.keys())
        boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            pose_nms(
                list_to_tensor(columns["bbox"]),
                list_to_tensor(columns["score"]),
                torch.tensor(list(range(len(columns["bbox"]))), dtype=torch.uint8),
                list_to_tensor(columns['keypoints']),
                list_to_tensor(columns['kp_score']),
                0)

        # 3) add result
        box_ids_lookup = columns["box_id"]
        for k, box_index in enumerate(ids):
            if isinstance(box_index, Iterable):
                box_id = box_ids_lookup[box_index[0]]  # lazy, just going to grab the first one.
            else:
                box_id = box_ids_lookup[box_index]
            pose = index[box_id]

            joints = preds_img[k]
            kp_score = preds_scores[k]
            keypoints = []
            for ii, joint in enumerate(joints):
                keypoints.append(Keypoint(joint[0].item(), joint[1].item(), kp_score[ii][0].item()))

            frame.poses.append(Pose2D(
                pose_id=str(uuid4()),
                track_label=str(int(pose["idx"].tolist()[0])),
                keypoints=keypoints,
                quality=pose["proposal_score"].tolist()[0],
                box_id=pose["box_id"],
                bbox=Box(boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0], boxes[k][3]-boxes[k][1]),
            ))

        frame_num = frame.image_name.split('.')[0]
        file_path = os.path.join(dirname, f"poses-{frame_num}.json")
        logging.info("writring to %s", file_path)
        with open(file_path, 'w') as fp:
            json.dump(dataclasses.asdict(frame), fp)
            fp.flush()
    except Exception as err:
        logging.exception("image_id %s nms went off the rails", poses[0]["image_id"])
