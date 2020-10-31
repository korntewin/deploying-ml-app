import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, Parallel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import uniform
from sklearn.preprocessing import PowerTransformer, StandardScaler

from lasso.preprocess import preprocess as pp
from lasso.preprocess import features
from lasso.config import config


prep_pipeline = Pipeline([
    ('replace_w_nan', pp.ReplaceQmarkToNan()),
    ('conv_to_one_cabin', pp.ConvertManyCabinsToOneCabin(config.CABIN_COLUMN)),
    ('get_title', pp.NameToTitle(config.NAME_COLUMN)),
    ('drop_features', pp.DropUnecessaryColumn(config.DROP_COLUMN)),
    ('reduce_cab_cardinality', pp.ReduceCardinality(config.CABIN_COLUMN)),
    ('fill_cat_vars', pp.FillCatVars(config.CAT_COLUMN, config.CAT_COLUMN_W_NA, 'Missing')),
    ('remove_rare_label', pp.RareLabelEncoder(config.CAT_COLUMN)),
    ('fill_num_vars', pp.FillNumVars(config.NUM_COLUMN, config.NUM_COLUMN_W_NA)),
    ('log_transformer', features.LogTransformer(config.CONT_NUM_COLUMN)),
    ('ordinal_encoder', pp.OrdinalEncoder(config.CAT_COLUMN)),
    ('scaler', pp.Scaler())
])


estimator_cv = RandomizedSearchCV(LogisticRegression(), config.PARAM_GRIDS, n_iter=config.N_ITER)


if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop(config.TARGET, axis=1), train[config.TARGET],
        test_size=0.2, random_state=42
        )

    prep_pipeline.fit(pd.concat([X_train, X_valid], axis=0), pd.concat([y_train, y_valid], axis=0))
    X_train_prep = prep_pipeline.transform(X_train)
