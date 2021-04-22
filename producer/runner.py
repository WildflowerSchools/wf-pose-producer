from datetime import datetime, timedelta
import gc
from itertools import chain, islice
import json
import logging
from multiprocessing import Process, Queue
import re
import time

import click
import torch
from video_io import fetch_videos
import dateparser

from producer.beta.monolith import PoseJob
from producer.helpers import chunks, json_dumps


LOG_FORMAT = '%(levelname)-8s %(asctime)s %(processName)-12s %(module)14s[%(lineno)04d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class PoseFactory:

    def __init__(
            self,
            environment_name,
            start_date,
            end_date,
            gpus="0",
            producer_batch_size=36,
            detection_batch_size=8,
            estimator_batch_size=10,
            debug=False,
            preload_only=False,
            worker_count=4,
        ):
        start = datetime.now()
        self.gpus = gpus.split(',')
        self.environment_name = environment_name
        self.start_date = start_date
        self.end_date = end_date
        self.producer_batch_size = producer_batch_size
        self.detection_batch_size = detection_batch_size
        self.estimator_batch_size = estimator_batch_size
        self.debug = debug
        self.worker_count = worker_count
        self.preload_only = preload_only
        if debug:
            logging.info("PoseFactory Starting")
            logging.info("environment_name:       %s", environment_name)
            logging.info("start_date:             %s", start_date)
            logging.info("end_date:               %s", end_date)
            logging.info("gpus:                   %s", gpus)
            logging.info("producer_batch_size:    %s", producer_batch_size)
            logging.info("detection_batch_size:   %s", detection_batch_size)
            logging.info("estimator_batch_size:   %s", estimator_batch_size)
            logging.info("debug:                  %s", debug)
            logging.info("worker_count:           %s", worker_count)
        if not preload_only:
            self.queue = Queue()
            self.process_list = [PoseProducer(self.queue, gpu, detection_batch_size, estimator_batch_size, debug, worker_count) for gpu in self.gpus]
            # use video-io to pull down videos and create job files
        try:
            self._fetch_videos()
        except Exception:
            logging.exception("problem with fetching videos")
        if not preload_only:
            # feed job file paths to the queue as they become available.
            # when all videos are ready and job files in queue the queue the `SHUTDOWN` commands and return.
            for _ in self.gpus:
                self.queue.put("SHUTDOWN")
            while not self.queue.empty() or self.queue.qsize() > len(self.process_list):
                time.sleep(1)
        end = datetime.now()
        logging.info("all work took %f seconds", (end - start).total_seconds())

    def _fetch_videos(self):
        results = fetch_videos(
            environment_name=self.environment_name,
            start=self.start_date,
            end=self.end_date,
            chunk_size=1000,
            local_video_directory='/data/prepared',
            video_filename_extension='mp4',
            download_workers=10,
        )
        logging.info('videos prepared, downloaded %s videos', len(results))
        # chunks = iter(lambda: tuple(islice(results, self.producer_batch_size)), ())
        # logging.info("%s chunks to process", len(list(chunks)))
        i = 0
        for chunk in chunks(list(results), self.producer_batch_size):
            i += 1
            fname = f'/work/{self.environment_name}-job-{i:06}.json'
            with open(fname, 'w') as outa_computown:
                for item in chunk:
                    outa_computown.write(json_dumps({
                        "timestamp": item["video_timestamp"],
                        "assignment_id": item["assignment_id"],
                        "environment_id": item["environment_id"],
                        "device_id": item["device_id"],
                        "path": item["video_local_path"],
                    }))
                    outa_computown.write('\n')
                outa_computown.flush()
            if not self.preload_only:
                self.queue.put(fname)


class PoseProducer:

    def __init__(self, queue, gpu, detection_batch_size=8, estimator_batch_size=10, debug=False, worker_count=4):
        self.gpu = gpu
        self.detection_batch_size = detection_batch_size
        self.estimator_batch_size = estimator_batch_size
        self.debug = debug
        self.worker_count = worker_count
        if self.debug:
            logging.info("gpu %s", gpu)
        self.p = Process(target=self, args=(queue, ))
        self.p.start()

    def __call__(self, queue):
        self.queue = queue
        failures = 0
        if self.debug:
            logging.info("PoseProducer %s started", self.gpu)
        while True:
            try:
                job = self.queue.get()
                if job == "SHUTDOWN":
                    if self.debug:
                        logging.info("PoseProducer %s will %s", self.gpu, job)
                    return
                if self.debug:
                    logging.info("PoseProducer %s found work: %s", self.gpu, len(job))
                    logging.info(job[0])
                    p_job = PoseJob(
                        job,
                        self.gpu,
                        self.detection_batch_size,
                        self.gpu,
                        self.estimator_batch_size,
                        worker_count=4,
                        debug=True,
                    )
                    p_job.start()
                    del p_job
                    gc.collect()
                    failures = 0
            except Exception as e:
                failures += 1
                logging.exception("problem?")
                if failures > 10:
                    logging.error("failures too much")
                    return
                time.sleep(1)


TIMEDELTA_REGEX = (r'((?P<days>-?\d+)d)?'
                   r'((?P<hours>-?\d+)h)?'
                   r'((?P<minutes>-?\d+)m)?')
TIMEDELTA_PATTERN = re.compile(TIMEDELTA_REGEX, re.IGNORECASE)


def parse_duration(time_str):
    parts = TIMEDELTA_PATTERN.match(time_str)
    assert parts is not None, "Could not parse any time information from '{}'.  Examples of valid strings: '8h', '2d8h5m20s', '2m4s'".format(time_str)
    time_params = {name: float(param) for name, param in parts.groupdict().items() if param}
    return timedelta(**time_params)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--environment-name', required=True, type=str)
@click.option('--start', required=True, type=str)
@click.option('--duration', required=False, type=str, default="8h")
@click.option('--producer-batch-size', required=False, type=int, default=36)
@click.option('--gpus', required=False, type=str, default="0")
@click.option('--detection-batch-size', required=False, type=int, default=8)
@click.option('--estimator-batch-size', required=False, type=int, default=10)
@click.option('--debug/--no-debug')
@click.option('--preload-only/--no-preload-only')
@click.option('--worker-count', required=False, type=int, default=4)
def main(environment_name, start, duration="8h", producer_batch_size=36, gpus="0", detection_batch_size=8, estimator_batch_size=10, debug=False, worker_count=4, preload_only=False):
    torch.multiprocessing.set_start_method('forkserver', force=True)
    start_date = dateparser.parse(start)  # example: "2021-02-09T07:30:00.000-0600"
    delta = parse_duration(duration)
    end_date = start_date + delta
    producer = PoseFactory(
        environment_name,
        start_date,
        end_date,
        gpus=gpus,
        producer_batch_size=producer_batch_size,
        detection_batch_size=detection_batch_size,
        estimator_batch_size=estimator_batch_size,
        debug=debug,
        worker_count=worker_count,
        preload_only=preload_only,
    )



if __name__ == '__main__':
    cli()
