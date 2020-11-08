import json

import pandas as pd
import numpy as np

from lasso.predict import make_prediction, make_prediction_proba
from lasso import config
from lasso.preprocess.data_management import load_dataset
from myapi import config as api_config
from myapi.logger_config import get_logger
from myapi.config import PREV_VER_PREDS_FILENAME

test_dataset = load_dataset(file_name=api_config.TEST_FILENAME)
_logger = get_logger(logger_name=__name__)


def capture_prev_ver_predictions():
    df = test_dataset[111:444].copy()
    # save to csv
    df.to_csv('prev_ver_dataset.csv', index=False)
    test_json = df.to_json(orient='records')
    test_preds = make_prediction_proba(input_data=test_json)

    test_pred = pd.DataFrame(test_preds['predictions'])
    test_pred_version = test_preds['version']

    test_pred.to_csv(api_config.PACKAGE_ROOT / api_config.PREV_VER_PREDS_FILENAME, 
    index=False)
    _logger.debug(f'save previous version {test_pred_version} \
        predictions: {api_config.PACKAGE_ROOT / api_config.PREV_VER_PREDS_FILENAME}')


if __name__ == '__main__':
    capture_prev_ver_predictions()