from flask import Flask

import api.controller


def create_app() -> None:
    flask_app = Flask('ml-app')

    flask_app.register_blueprint()

    return flask_app