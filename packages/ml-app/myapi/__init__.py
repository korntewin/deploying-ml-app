import logging

from myapi import config
from myapi.logger_config import get_logger


VERSION_FILENAME = config.PACKAGE_ROOT / 'VERSION'

_logger = get_logger(logger_name=__name__)

with open(VERSION_FILENAME, 'r') as version_file:
    _version_ = version_file.read().strip() 
    _logger.info(f'import version file: {_version_}')