from collections import namedtuple, defaultdict

import math
import numpy as np

EXCLUDE_THESE = ['class']

Entry = namedtuple('Entry', ['index', 'name', 'type', 'possible_values', 'value'])

def sigmoid(value):
    return 1 / (1 + np.exp(-value))

def sigmoidprime(value):
    return sigmoid(value) * (1 - sigmoid(value))

def what_is_my_bucket(data, attribute, bucket_params):
    buckets_per_continuous = bucket_params[0]
    inner_buckets = buckets_per_continuous - 2
    min_ = bucket_params[1][attribute]
    max_ = bucket_params[2][attribute]
    total_range = max_ - min_
    data_range = data - min_

    if inner_buckets <= 0:
        if data <= (min_ + max_)/2.0:
            return 0
        else:
            return 1

    if total_range == 0.0:
        scaled_range = 0.0
    else:
        scaled_range= data_range/total_range * inner_buckets
    # now values from 0 to buckets_per_continuous
    # then round down to the floor to get the integer buckets
    bucket = int(math.floor(scaled_range)) + 1
    if bucket <= 0:
        bucket = 0
    elif bucket > inner_buckets:
        bucket = inner_buckets + 1
    return bucket

class Result(object):
    def __init__(self, accuracy=None, precision=None, recall=None, numDataPoints=None, rocPredicted=None, rocActual=None):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.numDataPoints = numDataPoints
        self.rocPredicted = rocPredicted
        self.rocActual = rocActual

    def from_(self, results):
        total_data_points = 0
        accuracy_accum = 0
        precision_accum = 0
        recall_accum = 0
        rocPredicted = []
        rocActual = []
        for result in results:
            total_data_points += result.numDataPoints
            accuracy_accum += result.numDataPoints * result.accuracy
            precision_accum += result.numDataPoints * result.precision
            recall_accum += result.numDataPoints * result.recall
            rocPredicted.extend(result.rocPredicted)
            rocActual.extend(result.rocActual)

        self.accuracy = float(accuracy_accum) / total_data_points
        self.precision = float(precision_accum) / total_data_points
        self.recall = float(recall_accum) / total_data_points
        self.numDataPoints = total_data_points
        self.rocPredicted = rocPredicted
        self.rocActual = rocActual

        acc_stdev = 0.0
        pre_stdev = 0.0
        rec_stdev = 0.0
        for result in results:
            acc_stdev += math.pow(result.accuracy - self.accuracy, 2)
            pre_stdev += math.pow(result.precision - self.precision, 2)
            rec_stdev += math.pow(result.recall - self.recall, 2)

        self.acc_stdev = math.sqrt(acc_stdev / float(total_data_points))
        self.pre_stdev = math.sqrt(pre_stdev / float(total_data_points))
        self.rec_stdev = math.sqrt(rec_stdev / float(total_data_points))

        return self

    def print_output(self):
        tpbins = defaultdict(int)
        fpbins = defaultdict(int)
        # these are the counts for the total actual values for true and false
        total_true = 0
        total_false = 0
        for index in range(0, len(self.rocPredicted)):
            predicted = self.rocPredicted[index]
            actual = self.rocActual[index]
            # at what level does the predicted become tp or fp?
            # if actual is 1.0, then predicted is tp for all confidence levels below its value
            # if actual is 0.0, then predicted is fp for all confidence levels above its value 
            if actual == 0.0:
                total_false += 1
                fpbins[str(round(predicted - 0.005, 2))] += 1
            else:
                total_true += 1
                tpbins[str(round(predicted - 0.005, 2))] += 1

        area_accum = 0.0

        last_fp = 0.0
        accum_fp = 0
        last_tp = 0.0
        accum_tp = 0

        fps = []
        fps.append(last_fp)
        tps = []
        tps.append(last_tp)

        for reverse_index in np.arange(0.0, 1.0, 0.01):
            index = str(1.0 - reverse_index)
            accum_fp += fpbins[index]
            accum_tp += tpbins[index]
            this_fp = accum_fp / float(total_false)
            this_tp = accum_tp / float(total_true)

            delta_fp = this_fp - last_fp
            if delta_fp > 0.0:
                # add the trapezoid area
                area_accum += delta_fp * (this_tp + last_tp) / 2.0
            last_fp = this_fp + 0.0
            last_tp = this_tp + 0.0
            fps.append(last_fp)
            tps.append(last_tp)

        print('Result Report')
        if hasattr(self, 'acc_stdev'):
            print('Accuracy        %3.3f %3.3f' % (self.accuracy, self.acc_stdev,))
        else:
            print('Accuracy:       %3.3f' % (self.accuracy,))
        if hasattr(self, 'acc_stdev'):
            print('Precision:      %3.3f %3.3f' % (self.precision, self.pre_stdev,))
        else:
            print('Precision:      %3.3f' % (self.precision,))
        if hasattr(self, 'acc_stdev'):
            print('Recall:         %3.3f %3.3f' % (self.recall, self.rec_stdev,))
        else:
            print('Recall:         %3.3f' % (self.recall,))
        print('Area Under ROC: %3.3f' % (area_accum,))
        
class Datum(object):
    def __init__(self, item, schema, *args, **kwargs):
        super(Datum, self).__init__(*args, **kwargs)
        self.__data = {}
        self.schema = set()  # type: Set[str]
        for index, schematic in enumerate(schema):
            column_name = str(schematic)
            self.schema.update([column_name])
            column_type = 'CONTINUOUS'
            column_possible_values = None
            column_index = int(index)
            column_value = float(item[column_index])
            entry = Entry(column_index, column_name, column_type, column_possible_values, column_value)
            self.__data[column_name] = entry
            self.__data[column_index] = entry

    def get(self, name_or_index):
        return self.__data[name_or_index]

    def set(self, name, value):
        self.__data[name] = value
    
    def __hash__(self):
        vals = []
        for key in self.schema:
            vals.append((key, self.__get(key),))
            return hash(tuple(vals))

def defaultdict_gen2():
    return defaultdict(defaultdict_gen)

def defaultdict_gen():
    return defaultdict(int)

class NaiveBayes(object):
    def __init__(self, num_input_units, m_estimate, debug_output=False):
        self.num_input_units = num_input_units
        self.m = m_estimate

        self.pxy_cls = defaultdict(defaultdict_gen2)
        self.cls_count = defaultdict(int)
        self.total_count = 0

        self.debug_output = debug_output

        self.possible_values = defaultdict(int)

    def train(self, fold):
        schema = fold[0].schema
        for index, item in enumerate(fold):
            class_ = int(item.get('CLASS').value)
            self.cls_count[class_] += 1
            self.total_count += 1
            for category in schema:
                if category.lower() in EXCLUDE_THESE:
                    continue
                if item.get(category).type == 'CONTINUOUS':
                    print('unsupported category (not preprocessed from CONTINUOUS to buckets)')
                else:
                    if isinstance(item.get(category).possible_values, int):
                        self.possible_values[category] = item.get(category).possible_values
                    else:
                        self.possible_values[category] = len(item.get(category).possible_values)

                # index for nominal values
                # bins for continuous values
                value = item.get(category).value
                if isinstance(value, str):
                    # nominal
                    value = item.get(category).possible_values.index(value)
                elif item.get(category).type == 'CONTINUOUS':
                    print('unsupported cateogry CONTINUOUS') 

                self.pxy_cls[class_][category][value] += 1
        
    def log_likelihood(self, attribute, value, class_):
        likelihood = self.likelihood(attribute, value, class_)
        if likelihood == 0.0:
            return -float('inf')
        else:
            return math.log(likelihood)

    def likelihood(self, attribute, value, class_):
        print('likelihood(self, %s, %s, %s)' % (attribute, value, class_,))
        num_possible_values = self.possible_values[attribute]
        if num_possible_values == 0.0:
            print('bad attribute %s' % (attribute))
            import sys
            sys.exit(1)
        p = 1.0 / num_possible_values

        if (self.cls_count[class_] + self.m) == 0.0:
            print('bad cls_count %s' % (attribute,))
            return 0.0
        return (self.pxy_cls[class_][attribute][value] + self.m*p) / (self.cls_count[class_] + self.m)

    def positive_likelihood(self, datum, bucket_params):
        accum = 0.0
        for attribute in datum.schema:
            if attribute.lower() in EXCLUDE_THESE:
                continue
            value = datum.get(attribute).value
            if not isinstance(value, int):
                value = what_is_my_bucket(value, attribute, bucket_params)

            log_like = self.log_likelihood(attribute, datum.get(attribute).value, True)
            if log_like == -float('inf'):
                if self.debug_output:
                    print('early negative termination')
                return 0.0
            accum += log_like
        return accum

    def class_likelihood(self, datum, bucket_params, class_):
        accum = 0.0
        for attribute in datum.schema:
            if attribute.lower() in EXCLUDE_THESE:
                continue
            value = datum.get(attribute).value
            if not isinstance(value, int):
                value = what_is_my_bucket(value, attribute, bucket_params)

            log_likelihood = self.log_likelihood(attribute, datum.get(attribute).value, class_)
            if log_likelihood == -float('inf'):
                return 0.0
            accum += log_likelihood
        return accum

    def negative_likelihood(self, datum, bucket_params):
        accum = 0.0
        for attribute in datum.schema:
            if attribute.lower() in EXCLUDE_THESE:
                continue
            value = datum.get(attribute).value
            if not isinstance(value, int):
                value = what_is_my_bucket(value, attribute, bucket_params)
            log_like = self.log_likelihood(attribute, datum.get(attribute).value, False)
            if log_like == -float('inf'):
                if self.debug_output:
                    print('early negative termination')
                return 0.0
            accum += log_like
        return accum
    
    def classify(self, fold, bucket_params):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        actualValues = []
        predictedValues = []
        for datum in fold:
            # TODO convert this to a non-binary likelihood
            one_hot = []
            for class_ in self.cls_count:
                class_ = int(class_)
                one_hot.append(
                    # log_likelihood(self, attribute, value, class_)

                    self.class_likelihood(datum, bucket_params, class_) + 
                    math.log(self.cls_count[class_]) - 
                    math.log(self.total_count)
                    )

            max_index = 0
            max_value = one_hot[0]
            for index, val in enumerate(one_hot):
                if val > max_value:
                    max_index = index
                    max_value = val
            
            actualValues.append(float(datum.get('CLASS').value))
            predictedValues.append(max_index)

        accuracy = 0.0
        if self.total_count > 0:
            for actual, predicted in zip(actualValues, predictedValues):
                if actual == predicted:
                    accuracy += 1
            accuracy = float(accuracy)/self.total_count
        res = Result(accuracy=accuracy, precision=0.0, recall=0.0, numDataPoints=self.total_count, rocPredicted=[], rocActual=[])
        return res
