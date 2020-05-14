import os
import platform
import sys
import time

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






class AlphaPoser:

    def __init__(self, config_path, checkpoint, single_process=False, gpus=None, detbatch=5, posebatch=80, detector="tracker", qsize=1024, output_format="coco", output_indexed=False):
        #========================
        if torch.cuda.device_count():
            self.gpus = [-1]
        elif gpus is not None:
            self.gpus = [int(i) for i in gpus.split(',')]
        else:
            self.gpus = []
        self.device = torch.device("cuda:" + str(self.gpus[0]) if self.gpus[0] >= 0 else "cpu")
        self.detbatch = detbatch * len(self.gpus)
        self.posebatch = posebatch * len(self.gpus)
        self.tracking = (detector == 'tracker')
        self.detector = detector
        #========================
        self._single_process = single_process
        if not self._single_process:
            torch.multiprocessing.set_start_method('forkserver', force=True)
            torch.multiprocessing.set_sharing_strategy('file_system')
        #========================
        self.checkpoint = checkpoint
        self.config = update_config(config_path)
        self.qsize = qsize
        self.output_format = output_format
        self.output_indexed = output_indexed

    def process_video(self, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        det_loader = DetectionLoader(input_path, get_detector(self), self.config, self, batchSize=self.detbatch, mode="video").start()
        pose_model = builder.build_sppe(self.config.MODEL, preset_cfg=self.config.DATA_PRESET)

        print(f'Loading pose model from {self.checkpoint}...')
        pose_model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))

        if len(self.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=self.gpus).to(self.device)
        else:
            pose_model.to(self.device)
        pose_model.eval()

        # Init data writer
        writer = DataWriter(self.config, self, save_video=False, queueSize=self.qsize).start()

        data_len = det_loader.length
        # im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        try:
            print("inferring poses....")
            for i in data_len:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, os.path.basename(im_name))
                        continue
                    # Pose Estimation
                    inps = inps.to(self.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % self.posebatch:
                        leftover = 1
                    num_batches = datalen // self.posebatch + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * self.posebatch:min((j + 1) * self.posebatch, datalen)]
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    hm = hm.cpu()
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, os.path.basename(im_name))

            print_finish_info()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
            writer.stop()
            det_loader.stop()
        except KeyboardInterrupt:
            print_finish_info()
            # Thread won't be killed when press Ctrl+C
            if self.single_process:
                det_loader.terminate()
                while(writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...')
                writer.stop()
            else:
                # subprocesses are killed, manually clear queues
                writer.commit()
                writer.clear_queues()
                # det_loader.clear_queues()
        final_result = writer.results()
        write_json(final_result, self.outputpath, form=self.output_format, for_eval=self.output_indexed)
        print("Results have been written to json.")
