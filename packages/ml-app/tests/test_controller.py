import json

from api import config
from tests.conftest import flask_test_client
from lasso.preprocess.data_management import load_dataset
from lasso import __version__ as model_version
from api import __version__ as api_version


# def test_health_endpoint(flask_test_client):
#     response = flask_test_client.get('/health')

#     assert response.status_code == 200


def test_prediction_endpoint(flask_test_client):
    data = load_dataset(file_name=config.TEST_FILENAME)
    post_json = data[0:1].to_json(orient='records')
    response = flask_test_client.post('/v1/predict/lasso',
    json = post_json)

    assert response.status_code == 200
    response_json = json.loads(response.data)
    response_pred = response_json.get('predictions')
    response_version = response_json.get('version')
    assert response_pred <= 1 and response_pred >= 0
    assert response_version == model_version


def test_model_version_endpoint(flask_test_client):
    response = flask_test_client.get('/version')

    assert response.status_code == 200
    response_dic = json.loads(response.data)
    assert response_dic['api_version'] == api_version
    assert response_dic['model_version'] == model_version