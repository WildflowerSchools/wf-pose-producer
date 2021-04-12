from datetime import datetime, timedelta
import json
import logging
from uuid import uuid4

import cv2
import numpy as np
import torch
from alphapose.utils.config import update_config
from detector.apis import get_detector

from producer.helpers import now, ObjectView
from producer import settings
from producer.beta.boxtracking import check_for_output


class ImageExtractionWorker:

    def __init__(self, device):
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
        cfg = update_config("/data/alphapose-training/data/pose_cfgs/wf_alphapose_inference_config.yaml")
        self.detector = get_detector(args, cfg['DETECTOR'])

    def process(self, raw):
        obj = json.loads(raw)
        results = []
        path = obj.get("path")
        output_exists = check_for_output(path)
        logging.info("stareted %s", path)
        try:
            stream = cv2.VideoCapture(path)
            if stream.isOpened():
                frame_num = 0
                while True:
                    (grabbed, frame) = stream.read()
                    if not grabbed:
                        break
                    if f'poses-{frame_num}' not in output_exists:
                        img = self.detector.image_preprocess(frame)

                        if isinstance(img, np.ndarray):
                            img = torch.from_numpy(img)
                        # add one dimension at the front for batch if image shape (3,h,w)
                        if img.dim() == 3:
                            img = img.unsqueeze(0)

                        im_dim_list_k = frame.shape[1], frame.shape[0]

                        video_timestamp = datetime.strptime(obj["timestamp"], settings.ISO_FORMAT)
                        new_timestamp = video_timestamp + timedelta(seconds=frame_num * 0.1)

                        results.append({
                            "image_id": str(uuid4()),
                            "date": now(),
                            "im_name": str(frame_num) + '.jpg',
                            "img": img,
                            "orig_img": frame[:, :, ::-1],
                            "im_dim": im_dim_list_k,
                            "path": path,
                            "assignment_id": obj["assignment_id"],
                            "environment_id": obj["environment_id"],
                            "timestamp": new_timestamp.strftime(settings.ISO_FORMAT),
                        })
                    frame_num += 1
                    del frame
        except Exception as e:
            logging.exception(e)
        del stream
        return  results
