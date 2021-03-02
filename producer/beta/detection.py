import json
import logging
import os
import time

import click
import torch

from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from detector.apis import get_detector
import numpy as np

from producer.beta.loader import QueueWorkProcessor, ResultTarget
from producer.helpers import rabbit_params, packb, unpackb, columnarize, ObjectView
from producer.publisher import MonitorQueue




class ImageDetectionWorker(QueueWorkProcessor):

    def __init__(self, detector_cfg, detector_args, connection_params, source_queue_name, monitor_queue=None, result_queue=None, batch_size=2):
        super().__init__(connection_params, source_queue_name, monitor_queue=monitor_queue, result_queue=result_queue, batch_size=batch_size)
        self.detector_cfg = detector_cfg
        self.detector_args = detector_args
        self.detector = get_detector(detector_args, detector_cfg['DETECTOR'])
        self._input_size = detector_cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = detector_cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = detector_cfg.DATA_PRESET.SIGMA
        if detector_cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.detector_cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=detector_args.device)

    def process_batch(self, batch):
        columns = columnarize(batch, ["img", "orig_img", "im_name", "im_dim", "date", "path", "assignment_id", "environment_id", "timestamp"])
        imgs = columns["img"]
        orig_imgs = columns["orig_img"]
        im_names = columns["im_name"]
        im_dim_list = columns["im_dim"]
        date = columns["date"]
        path = columns["path"]
        assignment_id = columns["assignment_id"]
        environment_id = columns["environment_id"]
        timestamp = columns["timestamp"]
        with torch.no_grad():
            imgs = torch.cat(imgs)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        with torch.no_grad():
            # pad useless images to fill a batch, else there will be a bug
            for pad_i in range(self.batch_size - len(imgs)):
                imgs = torch.cat((imgs, torch.unsqueeze(imgs[0], dim=0)), 0)
                im_dim_list = torch.cat((im_dim_list, torch.unsqueeze(im_dim_list[0], dim=0)), 0)

            dets = self.detector.images_detection(imgs, im_dim_list)
            if isinstance(dets, int) or dets.shape[0] == 0:
                logging.info("nothing detected")
                return []
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

        results = []
        for k, oimg in enumerate(orig_imgs):
            boxes_k = boxes[dets[:, 0] == k]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                continue
            inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
            cropped_boxes = torch.zeros(boxes_k.size(0), 4)
            self.crop_images(boxes_k, oimg, inps, cropped_boxes)
            orig_img = oimg
            im_name = im_names[k]
            scores_k = scores[dets[:, 0] == k]
            ids_k = ids[dets[:, 0] == k]
            image_result = []
            for index, box in enumerate(boxes_k):
                image_result.append({
                    "score": scores_k[index],
                    "id": ids_k[index],
                    "inp": inps[index],
                    "cropped_box": cropped_boxes[index],
                    "box": box,
                })
            results.append(packb({
                "orig_img": orig_img,
                "im_name": im_name,
                "path": path[k],
                "date": date[k],
                "assignment_id": assignment_id[k],
                "environment_id": environment_id[k],
                "timestamp": timestamp[k],
                "boxes": image_result,
            }))
        return results

    def crop_images(self, boxes, orig_img, inps, cropped_boxes):
        with torch.no_grad():
            if boxes is None or boxes.nelement() == 0:
                return
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)


@click.command()
@click.option('--device', required=False, default="cpu")
@click.option('--batch', required=False, type=int, default=8)
@click.option('--monitor', required=False, default="detection")
@click.option('--monitor_limit', required=False, type=int, default=1000)
def main(device="cpu", batch=8, monitor="box-tracker", monitor_limit=1000):
    logging.info("options passed => device: %s  batch: %s  monitor: %s  monitor_limit: %s", device, batch, monitor, monitor_limit)
    gpus = []
    if device != "cpu":
        gpus = [int(device)]
        device = f"cuda:{device}"
    args = ObjectView({
        "sp": True,
        "tracking": "jde_1088x608",
        "detector": "yolov4",
        "device": device,
        "gpus": gpus,
    })
    monitor_queue = MonitorQueue(monitor, int(monitor_limit), 2)
    cfg = update_config("/data/alphapose-training/data/pose_cfgs/wf_alphapose_inference_config.yaml")
    worker = ImageDetectionWorker(cfg, args, rabbit_params(), 'detection', result_queue=ResultTarget('boxes', 'catalog'), batch_size=batch, monitor_queue=monitor_queue)
    worker.start()
    while True:
        time.sleep(5)


if __name__ == '__main__':
    main()
