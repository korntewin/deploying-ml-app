import pandas as pd
import numpy as np

from lasso import predict
from lasso.preprocess.data_management import load_dataset
from lasso.preprocess.data_validation import validate_data
from lasso.config import config
from lasso import predict


def test_predict():

    file_name = 'test.csv'
    data = load_dataset(file_name=file_name)
    test_json = data[0:1].to_json(orient='records')
    test_results = predict.make_prediction(input_data=test_json)

    assert test_results is not None
    assert isinstance(test_results['predictions'][0], np.int64)
    assert test_results['predictions'][0] <= 1
    assert test_results['predictions'][0] >= 0


def test_predict_proba():

    file_name = 'test.csv'
    data = load_dataset(file_name=file_name)
    test_json = data[0:1].to_json(orient='records')
    test_results = predict.make_prediction_proba(input_data=test_json)

    assert test_results is not None
    assert len(test_results['predictions'][0])==2
    assert test_results['predictions'][0][0] <= 1
    assert test_results['predictions'][0][0] >= 0

if __name__=='__main__':
    test_predict()