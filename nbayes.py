from bayesImpl import NaiveBayes, Result, Datum, Entry, what_is_my_bucket, EXCLUDE_THESE

import random
import datetime

from collections import deque
from random import shuffle

from math import sqrt, floor

import numpy as np

### Constants ###

EXCLUDE_THESE = ['class']

### Data Processing ###

def convert_to_old_representation(new_data):
    old_data = []

    new_data_data = new_data[0]
    new_data_labels = numpy.argmax(new_data, axis=1)
    for data_example, label in zip(new_data[0], new_data_labels):
        schema = list(range(0, len(data_example)))
        schema.append('CLASS')
        item = list(data_example)
        item.append(int(label))
        old_data.append(Datum(item, schema))

    return old_data




def set_to_buckets(parsed_data, buckets_per_continuous):
    standard_data = []
    schema = parsed_data[0].schema
    min_data = {}
    max_data = {}
    for item in parsed_data:
        for attribute in schema:
            if attribute.lower() in EXCLUDE_THESE:
                continue
            if item.get(attribute).type == 'CONTINUOUS':
                if attribute not in min_data:
                    min_data[attribute] = item.get(attribute).value
                min_data[attribute] = min(min_data[attribute], item.get(attribute).value)
                if attribute not in max_data:
                    max_data[attribute] = item.get(attribute).value
                max_data[attribute] = max(max_data[attribute], item.get(attribute).value)

    bucket_params = (buckets_per_continuous, min_data, max_data,)
    for item in parsed_data:
        for attribute in min_data:
            bucket = what_is_my_bucket(item.get(attribute).value, attribute, bucket_params)
            old_entry = item.get(attribute)
            item.set(attribute, Entry(old_entry.index, old_entry.name, 'BUCKET CONTINUOUS', buckets_per_continuous, bucket))

        for attribute in item.schema:
            if isinstance(item.get(attribute).value, str):
                old_entry = item.get(attribute)
                new_value = old_entry.possible_values.index(old_entry.value)
                item.set(attribute, Entry(old_entry.index, old_entry.name, 'BUCKET NOMINAL', len(old_entry.possible_values), new_value))
        standard_data.append(item)

    return standard_data, bucket_params

### Alg Runner ###

def run_algorithm(
    dataset,
    split=0.9,
    cross_validation=True,
    num_buckets=10,
    m_estimate=0,
    debug_output=False
    ):
    sample_length = 1.0
    if num_continuous_buckets <= 2:
        num_continuous_buckets = 2

    if debug_output:
        print('Begin parsing data')
        if cross_validation:
            dataset = load_wavs(split=None, sample_length=sample_length, cross_validation=5)
        else:
            dataset = load_wavs(split=split, sample_length=sample_length)
        print('Done parsing data')

    if cross_validation:
        new_folds = dataset._training
        old_folds = []
        for new_fold in new_folds:
            old_folds.append(convert_to_old_representation(new_fold))

        for test_fold, test_index in enumerate(old_folds):

            num_input_units = 0
            for item in old_training[0].schema:
                if item.lower() in EXCLUDE_THESE:
                    continue
                else:
                    num_input_units += 1

            nab  = NaiveBayes(num_input_units, m_estimate, debug_output)

            for training_fold, training_index in enumerate(old_folds):
                if test_index == training_index:
                    continue
                nab.train(training_fold)

            results_list.append(nab.classify(test_fold))

        total_result = Result().from_(results=results_list)
        total_result.print_output()







    else: # don't cross validate
        new_training = dataset._raw_training
        old_training = convert_to_old_representation(new_training)
        new_testing = dataset._testing
        old_testing = convert_to_old_representation(new_testing)
        data, bucket_params = set_to_buckets(old_training, num_continuous_buckets)

        num_input_units = 0
        for item in old_training[0].schema:
            if item.lower() in EXCLUDE_THESE:
                continue
            else:
                num_input_units += 1

        if debug_output:
            print('no cross validation')
        nab = NaiveBayes(num_input_units, m_estimate, debug_output)
        nab.train(old_training)
        result = nab.classify(old_testing, bucket_params)
        result.print_output()
