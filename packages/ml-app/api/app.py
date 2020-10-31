from flask import Flask

from api.logger_config import get_logger
from api import config

_logger = get_logger(logger_name=__name__)


def create_app(*, config_object:config.Config) -> None:
    flask_app = Flask('ml-app')
    flask_app.config.from_object(config_object)
    
    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)
    _logger.debug('Application instance is created')

    return flask_app