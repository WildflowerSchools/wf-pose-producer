import multiprocessing as mp
import logging
import os
import sys
import time

import msgpack
import nsq

from producer.alpha import AlphaPoser


class Agatho:

    def __init__(self, gpu="cuda0"):
        self.gpu = gpu
        self.start()

    def start(self):
        poser = AlphaPoser(
                    os.environ.get("ALPHAPOSE_CONFIG"),
                    os.environ.get("ALPHAPOSE_CHECKPOINT"),
                    gpu=self.gpu,
                    single_process=True,
                    output_format="cmu",
                )
        def handler(message):
            decoded = msgpack.unpackb(message.body, raw=False)
            logging.info(decoded)
            poser.process_video(video_path, outdir)
            message.finish()

        r = nsq.Reader(message_handler=handler,
                lookupd_http_addresses=['http://127.0.0.1:4161'],
                topic='wildflower', channel='videos', lookupd_poll_interval=2)
        nsq.run()


def start(gpu):
    Agatho(gpu)


def main(gpus):
    mp.set_start_method("spawn")
    procs = []
    print(f"starting {len(gpus)}")
    for gpu in gpus:
        print(gpu)
        p = mp.Process(name=f"agatho-{gpu}", target=start, args=(gpu,))
        p.start()
        procs.append(p)

    def joiner(p):
        p.join()

    map(joiner, procs)


if __name__ == '__main__':
    main(sys.argv[1:])
