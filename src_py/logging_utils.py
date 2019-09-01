import logging
import numpy as np
import os.path as path


class Logging(object):

    _logger = logging.getLogger(__name__)

    @classmethod
    def setup_custom_logger(cls, log_level):
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        cls._logger.setLevel(log_level.upper())
        cls._logger.addHandler(handler)
        cls._logger.info("Logging level set to {}".format(log_level.upper()))

    @classmethod
    def get_logger(cls):
        return cls._logger


class ResultStorage(object):

    def __init__(self, save_path):
        self.path = save_path

    def save(self, filename, data, silent=False):
        np.save(path.join(self.path, filename), data)
        if not silent:
            Logging.get_logger().log(logging.INFO, "{} {}".format(filename, data))
