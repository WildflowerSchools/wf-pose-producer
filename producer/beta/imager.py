from datetime import datetime, timedelta
import json
import logging
import os
import time

import cv2
import numpy as np
import torch
from detector.apis import get_detector

from producer.beta.loader import QueueWorkProcessor, ResultTarget
from producer.helpers import rabbit_params, now, packb, unpackb, ObjectView
from producer import settings as s


args = ObjectView({
    "sp": True,
    "tracking": "jde_1088x608",
    "detector": "yolov4",
    "device": "cuda:0",
    "gpus": [0],
})


class ImageExtractionWorker(QueueWorkProcessor):

    def __init__(self, cfg, detector_args, connection_params, source_queue_name, result_queue=None, batch_size=1, max_queue_size=10):
        super().__init__(connection_params, source_queue_name, result_queue=result_queue, batch_size=batch_size, max_queue_size=max_queue_size)
        self.detector = get_detector(detector_args, cfg['DETECTOR'])

    def prepare_single(self, message):
        return unpackb(message)

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


if __name__ == '__main__':
    from alphapose.utils.config import update_config
    cfg = update_config("/data/alphapose-training/data/pose_cfgs/wf_alphapose_inference_config.yaml")
    worker = ImageExtractionWorker(cfg, args, rabbit_params(), 'video', result_queue=ResultTarget('images', 'detector'))
    preloader, processor = worker.start()
    while not worker.stopped:
        time.sleep(5)
