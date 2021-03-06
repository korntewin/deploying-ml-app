import json

import pandas as pd
import numpy as np

import pytest

from tests.conftest import flask_test_client
from myapi.config import PACKAGE_ROOT, PREV_VER_PREDS_FILENAME, ACCETPTABLE_DIFF
from myapi.logger_config import get_logger
from lasso import config
from lasso.predict import make_prediction_proba
from lasso.preprocess.data_management import load_dataset

test_dataset = load_dataset(file_name=config.TEST_DATA_FN)
_logger = get_logger(logger_name=__name__)


@pytest.mark.differential
def test_differential(flask_test_client):
    prev_ver_test_pred = pd.read_csv(PACKAGE_ROOT / PREV_VER_PREDS_FILENAME)
    _logger.info('load previous version test predictions')

    df = test_dataset[111:444].copy()
    test_json = df.to_json(orient='records')
    _logger.info(f'Test differential Inputs: {test_json}')
    
    response = flask_test_client.post('/v1/predict_proba/lasso',
    json = test_json)
    pred_json = response.data

    pred_dic = json.loads(pred_json)
    cur_ver_test_pred = pred_dic['predictions']

    # test_preds = make_prediction_proba(input_data=test_json)
    # pd.DataFrame(test_preds['predictions']).to_csv('cur_ver_preds.csv', index=False)
    # cur_ver_test_pred = test_preds['predictions']

    _logger.info(f'Test differential Outputs: {cur_ver_test_pred}')

    assert len(prev_ver_test_pred) == len(cur_ver_test_pred)

    for prev_pred, cur_pred in zip(prev_ver_test_pred.values.tolist(), cur_ver_test_pred):
        assert np.isclose(prev_pred, cur_pred, rtol=ACCETPTABLE_DIFF).all()


