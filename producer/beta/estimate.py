import json
import logging
import os
import time

import click
import torch
from alphapose.models import builder
from alphapose.utils.writer import DataWriter
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import get_func_heatmap_to_coord
# from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.config import update_config

from producer.beta.loader import QueueWorkProcessor, ResultTarget
from producer.helpers import rabbit_params, packb, unpackb, columnarize, ObjectView, list_to_tensor
# from producer.beta.ppose import pose_nms


EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class PoseEstimationWorker(QueueWorkProcessor):

    def __init__(self, cfg, args, connection_params, source_queue_name, result_queue=None, batch_size=10):
        super().__init__(connection_params, source_queue_name, result_queue=result_queue, batch_size=batch_size)
        self.cfg = cfg
        self.args = args
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
        self.pose_model.to(args.device)
        self.pose_model.eval()
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    def process_batch(self, batch):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        results = []
        columns = columnarize(batch, ["inp", "orig_img", "im_name", "box", "score", "id", "cropped_box", "date", "path", "assignment_id", "environment_id", "timestamp", "image_id", "box_id"])
        inps = columns["inp"]
        orig_img = columns["orig_img"]
        im_name = columns["im_name"]
        image_ids = columns["image_id"]
        box_ids = columns["box_id"]
        boxes = columns["box"]
        scores = columns["score"]
        ids = columns["id"]
        cropped_boxes = columns["cropped_box"]
        date = columns["date"]
        path = columns["path"]
        assignment_id = columns["assignment_id"]
        environment_id = columns["environment_id"]
        timestamp = columns["timestamp"]

        logging.info("processing batch: inps[%s] im_name[%s] boxes[%s]", len(inps), len(im_name), len(boxes))
        try:
            boxes = list_to_tensor(boxes)
            inps = list_to_tensor(inps)
            scores = list_to_tensor(scores)
            ids = list_to_tensor(ids)
        except IndexError as e:
            logging.exception("empty lists for tensors")
            logging.info(columns)
            logging.info(batch)
            raise e
        with torch.no_grad():
            inps = inps.to(self.args.device)
            hm = self.pose_model(inps)
            hm_data = hm.cpu()
            if hm_data.size()[1] == 136:
                eval_joints = [*range(0, 136)]
            elif hm_data.size()[1] == 26:
                eval_joints = [*range(0, 26)]
            else:
                eval_joints = EVAL_JOINTS
            pose_coords = []
            pose_scores = []
            for i, bbox in enumerate(cropped_boxes):
                bbox = bbox.tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)
            for k, points in enumerate(preds_img):
                results.append(packb(
                    {
                        "image_id": image_ids[k],
                        "box_id": box_ids[k],
                        "imgname": im_name[k],
                        'keypoints': points,
                        'kp_score': preds_scores[k],
                        "score": scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx': ids[k],
                        'bbox': boxes[k],
                        "date": date[k],
                        "path": path[k],
                        "assignment_id": assignment_id[k],
                        "environment_id": environment_id[k],
                        "timestamp": timestamp[k],
                    }
                ))
        logging.info("processing batch: results[%s]", len(results))
        return results


@click.command()
@click.option('--device')
def main(device="cpu"):
    gpus = []
    if device != "cpu":
        gpus = [int(device)]
        device = f"cuda:{device}"
    args = ObjectView({
        "sp": True,
        "tracking": False, # "jde_1088x608"
        "detector": "yolov4",
        "checkpoint": "/build/AlphaPose/pretrained_models/alphapose-wf_res152_256x192.0.2.yolov4.pth",
        "qsize": 2048,
        "save_img": False,
        "pose_flow": False,
        "posebatch": 10,
        "outputpath": "/data/prepared",
        "format": "cmu",
        "eval": False,
        "min_box_area": 0,
        "device": device,
        "gpus": gpus,
    })
    cfg = update_config("/build/AlphaPose/data/pose_cfgs/wf_alphapose_inference_config.yaml")
    logging.info("preparing estimation worker")
    worker = PoseEstimationWorker(cfg, args, rabbit_params(), 'estimator', result_queue=ResultTarget('poses', '2dpose'))
    logging.info("starting estimation worker")
    worker.start()
    while True:
        time.sleep(5)


if __name__ == '__main__':
    main()
