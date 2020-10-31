from flask import Blueprint
from flask import Blueprint, request
from flask.json import jsonify

from api.logger_config import get_logger
from lasso.predict import make_prediction

_logger = get_logger(logger_name=__name__)
prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health() -> str:
    if request.method == 'GET':
        _logger.info('health status ok')
        return '200: OK'


@prediction_app.route('/v1/predict/lasso', methods=['POST'])
def make_predict() -> str:
    if request.method == 'POST':
        input_data = request.get_json()
        _logger.info(f'Inputs: {input_data}')

        results = make_prediction(input_data=input_data)
        _logger.info(f'Outputs: {results}')

        predictions = int(results.get('predictions')[0])
        version = results.get('version')

        return jsonify({
            'predictions': predictions,
            'version': version
        })

