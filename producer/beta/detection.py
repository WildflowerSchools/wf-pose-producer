import logging

import torch
import numpy as np

from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform
from alphapose.models import builder
from detector.apis import get_detector

from producer.helpers import ObjectView


class ImageDetector:

    def __init__(self, device, args):
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
        self.batch_size = int(args.get("batch_size", 1))
        self.detector_cfg = update_config("/build/AlphaPose/data/pose_cfgs/wf_alphapose_inference_config.yaml")
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

    def process_batch(self, inputs):
        imgs = []
        im_dim_list = []
        for frame in inputs:
            img = self.detector.image_preprocess(frame)
            oimg = frame[:, :, ::-1]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                img = img.unsqueeze(0)

            im_dim_list_k = frame.shape[1], frame.shape[0]
            if img is None:
                continue
            imgs.append(img)
            im_dim_list.append(im_dim_list_k)
        with torch.no_grad():
            imgs = torch.cat(dims)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

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
        for k, _ in enumerate(inputs):
            boxes_k = boxes[dets[:, 0] == k]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                continue
            scores_k = scores[dets[:, 0] == k]
            ids_k = ids[dets[:, 0] == k]
            image_result = []
            for index, box in enumerate(boxes_k):
                image_result.append({
                    "score": scores_k[index].tolist(),
                    "id": ids_k[index].tolist(),
                    # "cropped_box": cropped_boxes[index],
                    "box": box.tolist(),
                })
            results.append({
                "boxes": image_result,
            })
        return results

    def crop_images(self, boxes, orig_img, inps, cropped_boxes):
        with torch.no_grad():
            if boxes is None or boxes.nelement() == 0:
                return
            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)
