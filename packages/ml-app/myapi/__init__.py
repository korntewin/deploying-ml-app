import logging

from myapi import config
from myapi.logger_config import get_logger

VERSION_FILENAME = config.PACKAGE_ROOT / 'VERSION'


with open(VERSION_FILENAME, 'r') as version_file:
    __version__ = version_file.read().strip() 