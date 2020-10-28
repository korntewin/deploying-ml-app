import joblib

import pandas as pd
from sklearn.pipeline import Pipeline

from lasso.config import config


def load_dataset(*, file_name:str) -> pd.DataFrame:
    _data = pd.read_csv(config.DATASETS_DIR/file_name)
    return _data


def save_pipeline(*, pipeline) -> None: 
    saved_name = 'full_pipe_model'
    joblib.dump(pipeline, config.TRAIN_MODEL_DIR/config.MODEL_NAME)

    print("saved pipeline")


def load_pipeline(*, pipeline_name) -> Pipeline:
    pipeline = joblib.load(config.TRAIN_MODEL_DIR/pipeline_name)
    return pipeline