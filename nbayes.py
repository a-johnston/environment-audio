from bayesImpl import NaiveBayes, Result, Datum, Entry, what_is_my_bucket, EXCLUDE_THESE
from dataset import Dataset

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
    new_fft = new_data[0]
    label = np.argmax(new_data[1])
    schema = list(range(0, len(new_fft)))
    schema.append('CLASS')
    item = list(new_fft)
    item.append(int(label))
    return Datum(item, schema)

def set_to_buckets(parsed_data, num_buckets, bucket_params=None):
    standard_data = []
    schema = parsed_data[0].schema
    min_data = {}
    max_data = {}
    if bucket_params is None:
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

        bucket_params = (num_buckets, min_data, max_data,)
    for item in parsed_data:
        for attribute in min_data:
            bucket = what_is_my_bucket(item.get(attribute).value, attribute, bucket_params)
            old_entry = item.get(attribute)
            item.set(attribute, Entry(old_entry.index, old_entry.name, 'BUCKET CONTINUOUS', num_buckets, bucket))

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
    if num_buckets <= 2:
        num_buckets = 2

    if cross_validation:
        new_folds = dataset._training
        old_folds = [None] * len(new_folds)
        
        
        for index, new_fold in enumerate(new_folds):
            old_folds[index] = []
            # new_fold is a list of tuples of the format (data, label)
            for new_fold_set in new_fold:
                old_folds[index].append(convert_to_old_representation(new_fold_set))

        results_list = []

        for test_index, test_fold in enumerate(old_folds):
            num_input_units = 0
            
            for item in test_fold[0].schema:
                if item.lower() in EXCLUDE_THESE:
                    continue
                else:
                    num_input_units += 1

            nab  = NaiveBayes(num_input_units, m_estimate, debug_output)

            for training_index, training_fold in enumerate(old_folds):
                if test_index == training_index:
                    continue
                training_fold, bucket_params = set_to_buckets(training_fold, num_buckets)
                nab.train(training_fold)

            test_fold, unused_bucket_params = set_to_buckets(test_fold, num_buckets, bucket_params)
            results_list.append(nab.classify(test_fold, bucket_params))

        total_result = Result().from_(results=results_list)
        total_result.print_output()
        return total_result.accuracy







    else: # don't cross validate
        new_training = dataset._raw_training
        old_training = convert_to_old_representation(new_training)
        new_testing = dataset._testing
        old_testing = convert_to_old_representation(new_testing)
        data, bucket_params = set_to_buckets(old_training, num_buckets)

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
        return result.accuracy

if __name__ == '__main__':
    data_folder='datashort'
    split=0.9
    cross_validation=True
    sample_length = 1.0

    accuracy_accum = 0
    accuracy_count = 10

    for i in range(0, accuracy_count):
        if cross_validation:
            dataset = Dataset.load_wavs(data_folder=data_folder, split=None, sample_length=sample_length, cross_validation=5)
        else:
            dataset = Dataset.load_wavs(data_folder=data_folder, split=split, sample_length=sample_length)

        output = run_algorithm(
            dataset,
            split=split,
            cross_validation=cross_validation,
            num_buckets=40+2,
            m_estimate=2,
            debug_output=True)

        accuracy_accum += output

    print('%dx accuracy = %s' % (accuracy_count, (accuracy_accum/float(accuracy_count)),))
