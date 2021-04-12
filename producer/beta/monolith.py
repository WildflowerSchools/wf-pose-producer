import sys
sys.path.append("/opt/build/AlphaPose")

import concurrent.futures
from datetime import datetime
from itertools import chain, islice
import json
import logging
import multiprocessing.shared_memory
import os

import click
import torch

from producer.beta.imager import ImageExtractionWorker
from producer.beta.detection import ImageDetectionWorker
from producer.beta.boxtracking import BoxTracker, run_nms, drop_tombstone
from producer.beta.estimate import PoseEstimationWorker
from producer.helpers import chunks


class PoseJob:

    def __init__(
            self,
            job_batch,
            detection_device,
            detection_batch_size,
            estimator_device,
            estimator_batch_size,
            worker_count=4,
            debug=True,
        ):
        self.job_batch = job_batch
        self.detection_device = detection_device
        self.detection_batch_size = detection_batch_size
        self.estimator_device = estimator_device
        self.estimator_batch_size = estimator_batch_size
        self.worker_count = worker_count
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_count)
        self._load_names()
        self.__box_tracking = BoxTracker()
        self.debug = debug
        if debug:
            logging.info("=" * 80)
            logging.info(" job ready")
            logging.info(" detection_device:      %s", detection_device)
            logging.info(" detection_batch_size:  %s", detection_batch_size)
            logging.info(" estimator_device:      %s", estimator_device)
            logging.info(" estimator_batch_size:  %s", estimator_batch_size)
            logging.info(" job path count:        %s", len(self.names))
            logging.info(" workers:               %s", worker_count)
            logging.info("=" * 80)

    def _load_names(self):
        with open(self.job_batch, 'r') as fp:
            data = fp.read()
            # self.names = multiprocessing.shared_memory.ShareableList([item for item in data.split('\n') if len(item) > 0])
            self.names = [item for item in data.split('\n') if len(item) > 0]

    def cleanup(self):
        self.executor.shutdown()
        os.remove(self.job_batch)

    def start(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        start = datetime.now()
        logging.info("starting job")
        imager = ImageExtractionWorker("cpu")
        # extract all of the images.
        results = self.executor.map(imager.process, self.names)
        images = [item for item in chain.from_iterable(results) if item is not None]
        if len(images) == 0:
            return self.cleanup()
        self.__box_tracking.declare_images(images)
        logging.info("collected %s images", len(images))
        # create a generator for the detection batches
        chunker = chunks(images, self.detection_batch_size)
        del self.names
        logging.info("all images extracted")
        detector = ImageDetectionWorker(self.detection_device, self.detection_batch_size)
        logging.info("detector loaded")
        boxes = []
        batch_num = 0
        for batch in chunker:
            batch_num += 1
            if self.debug:
                logging.info("starting detection batch %s", batch_num)
            results = detector.process_batch(list(batch))
            # do box tracking
            results = self.__box_tracking.ingest_boxes(results)
            boxes.extend(results)
        if self.debug:
            logging.info("found some boxes  %s", len(boxes))
        # start estimation on the boxes
        del detector
        torch.cuda.empty_cache()
        chunker = chunks(boxes, self.estimator_batch_size)
        estimator = PoseEstimationWorker(self.estimator_device, self.estimator_batch_size)
        batch_num = 0
        for batch in chunker:
            batch_num += 1
            if self.debug:
                logging.info("starting estimation batch %s", batch_num)
            results = estimator.process_batch(list(batch))
            # reconcile poses
            self.__box_tracking.rectify_poses(results)
        # run NMS
        if self.debug:
            logging.info("estimation complete")
            # logging.info(self.__box_tracking.get_image_ids())
        # write frame data to output
        img_ids = self.__box_tracking.get_image_ids()
        pose_sets = []
        empty_images = []
        for image_id in img_ids:
            poses = self.__box_tracking.get_poses_for_image_id(image_id)
            if len(poses) > 0:
                pose_sets.append(poses)
            else:
                empty_images.append(self.__box_tracking.get_image_meta(image_id))
        # run_nms(pose_sets[0])
        self.executor.shutdown()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
        results = list(self.executor.map(run_nms, pose_sets))
        np_results = list(self.executor.map(drop_tombstone, empty_images))
        end = datetime.now()
        logging.info("work complete")
        del estimator
        torch.cuda.empty_cache()
        if self.debug:
            logging.info("clean")
            logging.info("frames with poses saved %s", len(results))
            logging.info("frames without poses saved %s", len(np_results))
            logging.info("-" * 80)
            logging.info("-" * 80)
            logging.info("processing took %f seconds", (end - start).total_seconds())
            logging.info("-" * 80)
            logging.info("-" * 80)
        return self.cleanup()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--job-batch', required=True, type=str)
@click.option('--detection-device', required=False, default="cpu")
@click.option('--detection-batch-size', required=False, type=int, default=8)
@click.option('--estimator-device', required=False, default="cpu")
@click.option('--estimator-batch-size', required=False, type=int, default=10)
@click.option('--debug/--no-debug')
@click.option('--worker-count', required=False, type=int, default=4)
def main(*args, **kwargs):
    LOG_FORMAT = '%(levelname)-8s %(asctime)s %(name)-10s %(funcName)-15s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    torch.multiprocessing.set_start_method('forkserver', force=True)
    job = PoseJob(*args, **kwargs)
    job.start()


if __name__ == '__main__':
    cli()
