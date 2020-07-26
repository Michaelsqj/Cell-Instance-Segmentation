import time
import logging


class show_time():
    def __init__(self, max_epoch):
        self.start_time = 0
        self.end_time = 0
        self.duration = 0
        self.epoch = 0
        self.max_epoch = max_epoch
        self.logger = logging.getLogger('show_time')

    def start(self, epoch):
        self.start_time = time.time()
        self.epoch = epoch

    def end(self):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        t = (self.max_epoch - self.epoch) * self.duration
        t += time.time()
        T = time.localtime(t)
        self.logger.info('etc:' + time.strftime("%Y-%m-%d %H:%M:%S", T))
