import typing as t
import json
from pprint import pprint
import copy

import pandas as pd
from marshmallow import fields, Schema, validate
from marshmallow import ValidationError
import numpy as np

from lasso.preprocess.data_management import load_dataset
from myapi import config


class DataSchema(Schema):
    passengerid = fields.Integer()
    pclass = fields.Integer()
    name = fields.String()
    sex = fields.String()
    age = fields.Float(allow_nan=True)
    sibsp = fields.Integer()
    parch = fields.Integer()
    ticket = fields.String()
    fare = fields.Float(allow_nan=True)
    cabin = fields.String(allow_none=True)
    embarked = fields.String(allow_none=True)


def _filter_error_inputs(errors: dict, input_data:t.List[dict]) \
    -> t.List[dict]:
    
    indexes = errors.keys()
    validated_data = copy.copy(input_data)

    for rm_ind in sorted(indexes, reverse=True):
        del validated_data[rm_ind]

    return validated_data


def validate_inputs(*, input_data:json):

    data_schema = DataSchema(many=True)
    input_dic = json.loads(input_data)

    errors = None
    try:
        results = data_schema.load(input_dic)
    except ValidationError as exc:
        errors = exc.messages

    # filter input
    if errors:
        validated_data = _filter_error_inputs(errors, input_dic)
        validated_data = json.dumps(validated_data)
        return validated_data, errors

    return input_data, errors


if __name__ == '__main__':
    data = load_dataset(file_name=config.TEST_FILENAME)
    data.iloc[0,1] = 'gg'
    post_json = data[0:2].to_json(orient='records')
    validate_inputs(input_data=post_json)
