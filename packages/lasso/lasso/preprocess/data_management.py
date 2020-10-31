import joblib
import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from lasso.config import config
from lasso import __version__ as _version


_logger = logging.getLogger(__name__)


def load_dataset(*, file_name:str) -> pd.DataFrame:
    _data = pd.read_csv(config.DATASETS_DIR/file_name)
    return _data


def save_pipeline(*, pipeline) -> None: 
    saved_name = f'{config.MODEL_NAME}{_version}.pkl'
    _logger.info(f'save {saved_name}')
    joblib.dump(pipeline, config.TRAIN_MODEL_DIR/saved_name)
    _logger.info(f'remove other pipeline')
    remove_pipeline(file_to_keep=saved_name)


def load_pipeline(*, pipeline_name) -> Pipeline:
    pipeline = joblib.load(config.TRAIN_MODEL_DIR/pipeline_name)
    return pipeline


def remove_pipeline(*, file_to_keep) -> None:

    for model in config.TRAIN_MODEL_DIR.iterdir():
        if model.name not in [file_to_keep, "__init__.py"]:
            model.unlink()
