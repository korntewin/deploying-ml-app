import logging
from logging import log

from lasso.config import config, logger_config


VERSION_PATH = config.ROOT_PATH/'VERSION'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logger_config.get_console_handler())
logger.propagate = False


with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()