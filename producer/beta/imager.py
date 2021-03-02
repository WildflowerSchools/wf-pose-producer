from datetime import datetime, timedelta
import json
import logging
import os
import time
from threading import Timer

import click
import cv2
import numpy as np
import torch
from alphapose.utils.config import update_config
from detector.apis import get_detector

from producer.beta.loader import QueueWorkProcessor, ResultTarget
from producer.helpers import rabbit_params, now, packb, unpackb, ObjectView
from producer import settings as s
from producer.publisher import MonitorQueue


class ImageExtractionWorker(QueueWorkProcessor):

    def __init__(self, cfg, detector_args, connection_params, source_queue_name, monitor_queue=None, result_queue=None, batch_size=1):
        super().__init__(connection_params, source_queue_name, result_queue=result_queue, batch_size=batch_size, monitor_queue=monitor_queue)
        self.detector = get_detector(detector_args, cfg['DETECTOR'])

    def process_batch(self, batch):
        results = []
        for obj in batch:
            path = obj.get("path")
            logging.info("stareted %s", path)
            stream = cv2.VideoCapture(path)
            assert stream.isOpened(), 'Cannot capture source'
            frame_num = 0
            while True:
                (grabbed, frame) = stream.read()
                if not grabbed:
                    break
                img = self.detector.image_preprocess(frame)

                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                # add one dimension at the front for batch if image shape (3,h,w)
                if img.dim() == 3:
                    img = img.unsqueeze(0)

                im_dim_list_k = frame.shape[1], frame.shape[0]

                video_timestamp = datetime.strptime(obj["timestamp"], s.ISO_FORMAT)
                new_timestamp = video_timestamp + timedelta(seconds=frame_num * 0.1)

                results.append(packb({
                    "date": now(),
                    "im_name": str(frame_num) + '.jpg',
                    "img": img,
                    "orig_img": frame[:, :, ::-1],
                    "im_dim": im_dim_list_k,
                    "date": now(),
                    "path": obj["path"],
                    "assignment_id": obj["assignment_id"],
                    "environment_id": obj["environment_id"],
                    "timestamp": new_timestamp.strftime(s.ISO_FORMAT),
                }))
                frame_num += 1
        return  results


@click.command()
@click.option('--device', required=False, default="cpu")
@click.option('--batch', required=False, type=int, default=8)
@click.option('--monitor', required=False, default="detection")
@click.option('--monitor_limit', required=False, type=int, default=1000)
def main(device="cpu", batch=8, monitor="detection", monitor_limit=1000):
    logging.info("options passed => device: %s  batch: %s  monitor: %s  monitor_limit: %s", device, batch, monitor, monitor_limit)
    cfg = update_config("/data/alphapose-training/data/pose_cfgs/wf_alphapose_inference_config.yaml")
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
    monitor_queue = MonitorQueue(monitor, int(monitor_limit), 5)
    worker = ImageExtractionWorker(cfg, args, rabbit_params(), 'video', monitor_queue=monitor_queue, result_queue=ResultTarget('images', 'detector'), batch_size=int(batch))
    worker.start()
    while True:
        time.sleep(5)


if __name__ == '__main__':
    main()
