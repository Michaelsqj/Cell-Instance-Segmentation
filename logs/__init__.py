import logging, logging.config


def build_logger(fname):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(fname, mode='a')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger('main')
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
