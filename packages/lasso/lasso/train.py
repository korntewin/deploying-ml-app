import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from lasso.config import config
from lasso.preprocess.data_management import load_dataset, save_pipeline
from lasso import pipeline
from lasso import __version__ as _version


_logger = logging.getLogger(__name__)


def run_training() -> None:
    '''better train (implementing cv to avoid losing the data for validation'''

    # divide dataset
    data = load_dataset(file_name=config.TRAIN_DATA_FN)

    X_train = data[config.TRAIN_FEATURES].sample(frac=1, random_state=42).drop(columns=config.TARGET)
    y_train = data[config.TRAIN_FEATURES].sample(frac=1, random_state=42)[config.TARGET[0]]
    pipeline.prep_pipeline.fit(X_train, y_train)

    # prep input
    prep_train = pipeline.prep_pipeline.transform(X_train)

    # fit
    np.random.seed(42)
    pipeline.estimator_cv.fit(prep_train, y_train)

    full_pipeline = Pipeline([
        ('prep_pipeline', pipeline.prep_pipeline),
        ('estimator', pipeline.estimator_cv.best_estimator_)
    ])

    # print reulsts
    print(f'best estimator: {pipeline.estimator_cv.best_estimator_}')
    print(f'full pipe score: {pipeline.estimator_cv.best_score_}')

    _logger.info(f"save model version: {_version}")
    save_pipeline(pipeline=full_pipeline)


if __name__ == '__main__':
    run_training()
    