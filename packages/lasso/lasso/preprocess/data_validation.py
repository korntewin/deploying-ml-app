'''
data validation module
'''

import numpy as np

from lasso.config import config


def validate_data(input_data):
    
    valid_input_data = input_data[input_data[config.CAT_COLUMN_NA_NOT_ALLOW
            + config.NUM_COLUMN_NA_NOT_ALLOW].notnull().all(axis=1)]

    return valid_input_data