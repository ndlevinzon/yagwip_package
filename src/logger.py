import logging
import os
from datetime import datetime

def setup_logger(debug_mode=False):
    logger = logging.getLogger("GROLEAP")
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger
