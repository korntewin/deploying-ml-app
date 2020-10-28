import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from lasso.config import config
from lasso.preprocess.data_management import load_dataset, save_pipeline
from lasso import pipeline


def run_training() -> None:

    '''conventional train (the data is splitted to train and valid set)'''
    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     train.drop(config.TARGET, axis=1), train[config.TARGET[0]],
    #     test_size=0.2, random_state=42
    #     )

    # pipeline.prep_pipeline.fit(pd.concat([X_train, X_valid], axis=0), pd.concat([y_train, y_valid], axis=0))

    # # preprocessing input
    # prep_train = pipeline.prep_pipeline.transform(X_train) 
    
    # # fit
    # np.random.seed(42)
    # pipeline.estimator_cv.fit(prep_train, y_train)

    # # save prep pipeline and estimator
    # full_pipeline = Pipeline([
    #     ('prep_pipeline', pipeline.prep_pipeline),
    #     ('estimator', pipeline.estimator_cv.best_estimator_)
    # ])

    # # print reulsts
    # print(f'best estimator: {pipeline.estimator_cv.best_estimator_}')
    # print(f'full pipe score: {full_pipeline.score(X_train, y_train)}')

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

    save_pipeline(pipeline=full_pipeline)


if __name__ == '__main__':
    run_training()
    