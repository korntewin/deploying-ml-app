from scipy.stats import uniform
import lasso # note that, the lasso path will be specify in tox.ini
import pathlib
import numpy as np


ROOT_PATH = pathlib.Path(lasso.__file__).resolve().parent
TRAIN_MODEL_DIR = ROOT_PATH/'trained_models'
DATASETS_DIR = ROOT_PATH/'datasets'


# data
TRAIN_DATA_FN = 'train.csv'
MODEL_NAME = 'full_pipe_model_v'

# variables
CABIN_COLUMN = 'cabin'
NAME_COLUMN = 'name'
DROP_COLUMN = ['name','ticket', 'boat', 'body','home.dest']
CAT_COLUMN = ['sex', 'cabin', 'embarked', 'title']
NUM_COLUMN = ['pclass', 'age', 'sibsp', 'parch', 'fare']
CONT_NUM_COLUMN = ['age', 'fare']
CAT_COLUMN_W_NA = ['cabin', 'embarked']
NUM_COLUMN_W_NA = ['age', 'fare']

NUM_COLUMN_NA_NOT_ALLOW = [col for col in NUM_COLUMN if col not in NUM_COLUMN_W_NA]
CAT_COLUMN_NA_NOT_ALLOW = [col for col in CAT_COLUMN if col not in CAT_COLUMN_W_NA]
CAT_COLUMN_NA_NOT_ALLOW.remove('title')

TRAIN_FEATURES = list(set(DROP_COLUMN).union(set(CAT_COLUMN)).union(set(NUM_COLUMN)))+['survived']
TRAIN_FEATURES.remove('title')

FEATURES = [
    'pclass',
    'name',
    'sex',
    'age',
    'sibsp',
    'parch',
    'ticket',
    'fare',
    'cabin',
    'embarked'
]

TARGET = ['survived']


# model parameter
PARAM_GRIDS = {'C':uniform(0.0001, 10)}
N_ITER = 50


if __name__ == '__main__':
    print(pathlib.Path(lasso.__file__).resolve())
