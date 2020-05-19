import logging
import os
import platform
import sys
import time

from gpu_utils import gpu_init
import numpy as np
import torch

from detector.apis import get_detector
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.writer import DataWriter

from producer.helpers import output_json_exists


class AlphaPoser:

    def __init__(self, config_path, checkpoint, single_process=False, detbatch=5, posebatch=80, gpu=None,
                 detector="yolo", qsize=1024, output_format="coco", output_indexed=False, pose_track=False):
        #========================
        self.sp = self._single_process = single_process
        if not self._single_process:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')
        #========================
        self.checkpoint = checkpoint
        self.config = update_config(config_path)
        self.qsize = qsize
        self.output_format = self.format = output_format
        self.output_indexed = output_indexed
        self.save_img = False
        self.pose_track = pose_track
        self.debug = False
        self.min_box_area = 0
        self.vis = False
        #========================
        logging.info(f"cuda device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() < 1:
            self.gpus = []
        elif gpu is not None:
            self.gpus = [int(gpu)]
        else:
            self.gpus = [gpu_init(best_gpu_metric="mem")]
        if len(self.gpus):
            self.device = torch.device("cuda:" + str(self.gpus[0]))
        else:
            self.device = torch.device("cpu")
        self.detbatch = detbatch  #  * len(self.gpus)
        self.posebatch = posebatch  #  * len(self.gpus)
        self.tracking = (detector == 'tracker')
        self.detector = detector
        self.pose_model =  builder.build_sppe(self.config.MODEL, preset_cfg=self.config.DATA_PRESET)
        logging.info(f'Loading pose model from {self.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.pose_model.to(self.device)
        self.pose_model.eval()
        self.detector_instance = get_detector(self)

    def process_video(self, input_path, output_path):
        if output_json_exists(output_path):
            logging.info("output exists, skipping")
            return
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.outputpath = output_path
        det_loader = DetectionLoader(input_path, self.detector_instance, self.config, self, batchSize=self.detbatch, mode="video").start()
        # Init data writer
        writer = DataWriter(self.config, self, save_video=False, queueSize=self.qsize).start()

        data_len = det_loader.length
        try:
            logging.info("inferring poses....")
            for i in range(data_len):
                logging.info(f"{i} of {data_len} starting")
                with torch.no_grad():
                    logging.info("detecting")
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        logging.info("orig_img is None, must be done")
                        break
                    if boxes is None or boxes.nelement() == 0:
                        logging.info("no poses in frame")
                        writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue
                    # Pose Estimation
                    logging.info("estimating poses")
                    inps = inps.to(self.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % self.posebatch:
                        leftover = 1
                    num_batches = datalen // self.posebatch + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * self.posebatch:min((j + 1) * self.posebatch, datalen)]
                        hm_j = self.pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    logging.info("moving to CPU")
                    hm = hm.cpu()
                    logging.info("saving")
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))
                    del hm
                logging.info(f"{i} of {data_len} complete")

            logging.info("finished inference")

            while(writer.running()):
                time.sleep(1)
                logging.info('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
            det_loader.stop()
        except KeyboardInterrupt:
            # logging.info_finish_info()
            # Thread won't be killed when press Ctrl+C
            if self._single_process:
                det_loader.terminate()
                while(writer.running()):
                    time.sleep(1)
                    logging.info('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
                writer.stop()
            else:
                # subprocesses are killed, manually clear queues
                writer.commit()
                writer.clear_queues()
                # det_loader.clear_queues()
        final_result = writer.results()
        write_json(final_result, self.outputpath, form=self.output_format, for_eval=self.output_indexed)
        logging.info("Results have been written to json.")
        del det_loader
        del writer
        del final_result
