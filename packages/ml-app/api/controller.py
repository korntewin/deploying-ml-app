import json

from flask import Blueprint
from flask import Blueprint, request
from flask.json import jsonify

import api
from api.logger_config import get_logger
from api.validation import validate_inputs
from lasso.predict import make_prediction
from api import __version__ as api_version
from lasso import __version__ as model_version

_logger = get_logger(logger_name=__name__)
prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health() -> str:
    if request.method == 'GET':
        _logger.info('health status ok')
        return '200: OK'


@prediction_app.route('/v1/predict/lasso', methods=['POST'])
def make_predict() -> json:
    if request.method == 'POST':
        input_data = request.get_json()
        _logger.info(f'Inputs: {input_data}')

        input_data, errors = validate_inputs(input_data=input_data)
        _logger.info(f'Validate Inputs: {input_data}')

        results = make_prediction(input_data=input_data)
        _logger.info(f'Outputs: {results}')

        predictions = int(results.get('predictions')[0])
        version = results.get('version')

        return jsonify({
            'predictions': predictions,
            'version': version,
            'errors': errors
        })


@prediction_app.route('/version', methods=['GET'])
def get_version() -> json:
    if request.method == 'GET':
        return jsonify({
            'api_version':api_version,
            'model_version': model_version
        })

