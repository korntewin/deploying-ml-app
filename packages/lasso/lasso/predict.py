import joblib
import json
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from lasso.config import config
from lasso.preprocess.data_management import load_pipeline, load_dataset
from lasso.preprocess.data_validation import validate_data
from lasso import __version__ as _version


model_name = f"{config.MODEL_NAME}{_version}.pkl"
_logger = logging.getLogger(__name__)
pipeline = load_pipeline(pipeline_name=model_name)


def make_prediction(*, input_data:json) -> dict:
    data = pd.read_json(input_data)
    data = validate_data(data)

    pred = pipeline.predict(data[config.FEATURES])
    _logger.info(
        f"Making prediction with model version: {_version} "
        f"Inputs: {data} "
        f"Predictions: {pred}"
    )

    response = {'predictions': pred.tolist(), "version": _version}
    return response


def make_prediction_proba(*, input_data) -> dict:
    data = pd.read_json(input_data)
    pred = pipeline.predict_proba(data[config.FEATURES])
    response = {'predictions': pred.tolist(), "version":_version}
    return response


if __name__ == '__main__':
    
    data = load_dataset(file_name=config.TRAIN_DATA_FN)

    X_train = data[config.TRAIN_FEATURES].sample(frac=1, random_state=42).drop(columns=config.TARGET)
    y_train = data[config.TRAIN_FEATURES].sample(frac=1, random_state=42)[config.TARGET[0]]

    # load model
    print(pipeline.score(X_train, y_train))