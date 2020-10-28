import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from lasso.config import config
from lasso.preprocess.data_management import load_pipeline


model_name = config.MODEL_NAME
pipeline = load_pipeline(pipeline_name=model_name)


def predict(*, input_data) -> dict:
    data = pd.read_json(input_data)
    pred = pipeline.predict(data[config.FEATURES])
    response = {'predictions': pred}
    return response


def predict_proba(*, input_data) -> dict:
    data = pd.read_json(input_data)
    pred = pipeline.predict_proba(data[config.FEATURES])
    response = {'predictions': pred}
    return response


# if __name__ == '__main__':
    
#     train = pd.read_csv(cf.TRAIN_DATA_FN)

#     X_train, X_valid, y_train, y_valid = train_test_split(
#         train.drop(cf.TARGET, axis=1), train[cf.TARGET[0]],
#         test_size=0.2, random_state=42
#         )
    
#     # load model
#     print(predict_proba(X_valid))