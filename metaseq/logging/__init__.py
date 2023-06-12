import logging
import os
import sys
import time
from typing import List


def get_logger(name: str, logger_blocklist: List[str] = []):
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
    logger = logging.getLogger(name)

    return logger
