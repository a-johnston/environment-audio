#!/usr/bin/env python3
import sys
from model import *
from dataset import *
from time import time
import json


def run(
    model,
    dataset,
    iterations=50000,
    print_every=1000,
):
    """Trains and evaluates the given model using the given dataset. Prints out
       progress and accuracy at a set interval as well as final accuracy.
    """
    print('Training ' + model.__class__.__name__ + '\n')

    start_time = time()

    for i in range(iterations):
        training_data = dataset.training()
        model.train(training_data[0], training_data[1])

        if print_every and i % print_every == 0:
            complete = i / iterations
            test_data = dataset.testing()
            accuracy = model.accuracy(test_data[0], test_data[1])

            print_args = (complete * 100, i, iterations, accuracy * 100, time() - start_time)

            print(
                'Training phase %.2f%% complete.\t(%d/%d)\tAccuracy: %6.2f%%\tElapsed: %.2fs'
                % print_args
            )

    test_data = dataset.testing()
    accuracy = model.accuracy(test_data[0], test_data[1])
    print('\n' + '-' * 85)
    print(
        'Training complete.\t\t\t\tAccuracy: %6.2f%%\tElapsed %.2fs'
        % (accuracy * 100, time() - start_time)
    )


def __try_number(value):
    try:
        f = float(value)
        i = int(f)
        return i if i == f else f
    except:
        return value


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ./trainer.py model data/mock [*args] **kwargs')
        print('Model options: ' + ', '.join(Model.models.keys()))
        sys.exit(1)

    args = []
    kwargs = {}

    if len(sys.argv) > 3:
        start = 3
        if sys.argv[3].startswith('[') and sys.argv[3].endswith(']'):
            start = 4
            args = json.loads(sys.argv[3])
        for i in range(start, len(sys.argv)):
            key, value = sys.argv[i].split('=')
            kwargs[key] = __try_number(value)

    model = sys.argv[1]
    if model not in Model.models:
        print('Unknown model ' + model + '.')
        print('Model options: ' + ', '.join(Model.models.keys()))
        sys.exit(1)

    if sys.argv[2] == 'mock':
        dataset = Dataset.mock()
    else:
        dataset = Dataset.load_wavs(sys.argv[2])
    model = Model.models[model](dataset, *args, **kwargs)

    run(model, dataset)
