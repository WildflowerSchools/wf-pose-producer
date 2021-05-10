import logging

import torch
import numpy as np

from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from detector.apis import get_detector

from producer.helpers import columnarize, ObjectView


class ImageDetectionWorker:

    def __init__(self, device, batch_size=2):
        gpus = []
        if device != "cpu":
            gpus = [int(device)]
            device = f"cuda:{device}"
        self.detector_args = ObjectView({
            "sp": True,
            "tracking": "jde_1088x608",
            "detector": "yolov4",
            "device": device,
            "gpus": gpus,
        })
        self.batch_size = batch_size
        self.detector_cfg = update_config("/data/alphapose-training/data/pose_cfgs/wf_alphapose_inference_config.yaml")
        self.detector = get_detector(self.detector_args, self.detector_cfg['DETECTOR'])
        self._input_size = self.detector_cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = self.detector_cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = self.detector_cfg.DATA_PRESET.SIGMA
        if self.detector_cfg.DATA_PRESET.TYPE == 'simple':
            pose_dataset = builder.retrieve_dataset(self.detector_cfg.DATASET.TRAIN)
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.detector_args.device)

    def process_batch(self, batch):
        columns = columnarize(batch, ["img", "image_id", "orig_img", "im_name", "im_dim", "date", "path", "assignment_id", "environment_id", "timestamp"])
        imgs = columns["img"]
        orig_imgs = columns["orig_img"]
        im_names = columns["im_name"]
        im_dim_list = columns["im_dim"]
        date = columns["date"]
        path = columns["path"]
        assignment_id = columns["assignment_id"]
        environment_id = columns["environment_id"]
        timestamp = columns["timestamp"]
        image_id = columns["image_id"]
        with torch.no_grad():
            imgs = torch.cat(imgs)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        with torch.no_grad():
            # pad useless images to fill a batch, else there will be a bug
            for _ in range(self.batch_size - len(imgs)):
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
            results.append({
                "image_id": image_id[k],
                "orig_img": orig_img,
                "im_name": im_name,
                "path": path[k],
                "date": date[k],
                "assignment_id": assignment_id[k],
                "environment_id": environment_id[k],
                "timestamp": timestamp[k],
                "boxes": image_result,
            })
        del batch
        del columns
        return results

    def crop_images(self, boxes, orig_img, inps, cropped_boxes):
        with torch.no_grad():
            if boxes is None or boxes.nelement() == 0:
                return
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)
