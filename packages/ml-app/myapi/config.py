import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PREV_VER_PREDS_FILENAME = 'prev_version_preds.csv'
TEST_FILENAME = 'test.csv'
ACCETPTABLE_DIFF = 0.05


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'this-really-needs-to-be-changed'
    SERVER_PORT = 5000


class ProductionConfig(Config):
    DEBUG = False
    SERVER_PORT = os.environ.get('PORT', 5000)


class DevelopmentConfig(Config):
    DEBUG = True
    DEVELOPMENT = True


class TestingConfig(Config):
    TESTING = True