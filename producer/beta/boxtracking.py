import json
import logging
import os
import time
from uuid import uuid4

import redis

from producer.beta.loader import QueueWorkProcessor, ResultTarget
from producer.helpers import rabbit_params, now, packb, unpackb


class BoxTrackerWorker(QueueWorkProcessor):

    def __init__(self, connection_params, source_queue_name, result_queue=None, batch_size=20, max_queue_size=5):
        super().__init__(connection_params, source_queue_name, result_queue=result_queue, batch_size=batch_size, max_queue_size=max_queue_size)

    def prepare_single(self, message):
        return unpackb(message)

    def process_batch(self, batch):
        redis_conn = redis.Redis(host="redis")
        results = []
        for image in batch:
            image_id = str(uuid4())
            base = {
                "image_id": image_id,
                "orig_img": image["orig_img"],
                "im_name": image["im_name"],
                "path": image["path"],
                "date": image["date"],
                "assignment_id": image["assignment_id"],
                "environment_id": image["environment_id"],
                "timestamp": image["timestamp"],
            }
            box_ids = []
            for box in image["boxes"]:
                box.update(base)
                box_id = str(uuid4())
                box["box_id"] = box_id
                box_ids.append(box_id)
                results.append(packb(box))
            redis_conn.sadd(f"input.{image_id}.manifest", *box_ids)
        return results


if __name__ == '__main__':
    from alphapose.utils.config import update_config
    cfg = update_config("/data/alphapose-training/data/pose_cfgs/wf_alphapose_inference_config.yaml")
    worker = BoxTrackerWorker(rabbit_params(), 'box-tracker', result_queue=ResultTarget('boxes', 'estimation'))
    preloader, processor = worker.start()
    while not worker.stopped:
        time.sleep(5)
