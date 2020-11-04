import logging
from logging.handlers import TimedRotatingFileHandler
from logging import StreamHandler
from logging import Logger
import os
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'

FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(funcName)s:%(lineno)d — %(message)s"
)


def get_console_handler() -> StreamHandler:
    handler = StreamHandler(sys.stdout)
    handler.setFormatter(FORMAT)
    return handler


def get_file_handler() -> TimedRotatingFileHandler:
    handler = TimedRotatingFileHandler(
        filename=LOG_FILE, when='midnight')
    handler.setFormatter(FORMAT)
    handler.setLevel(logging.WARNING)
    return handler


def get_logger(*, logger_name: str) -> Logger:
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(get_console_handler())
    _logger.addHandler(get_file_handler())

    return _logger
