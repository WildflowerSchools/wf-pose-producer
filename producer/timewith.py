import logging
import time



class timewith():
    def __init__(self, msg):
        self.measurement = msg
        self.last_check = None
        self.last_checkpoint_name = None
        self.start = self.last_check = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    def checkpoint(self, name):
        self.last_checkpoint_name = name
        elapsed = (self.elapsed * 1000.0)
        now = time.time()
        interval_time = (now - self.last_check) * 1000.0
        self.last_check = now
        logging.info('%s %s took %f ms (%s since last checkpoint)', self.measurement, name, elapsed, interval_time)
        return elapsed

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
