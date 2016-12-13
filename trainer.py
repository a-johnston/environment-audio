#!/usr/bin/env python3
import sys
from model import *
from dataset import *
from time import time
import json


def run(
    model,
    dataset,
    iterations=5000,
    print_every=50,
):
    """Trains and evaluates the given model using the given dataset. Prints out
       progress and accuracy at a set interval as well as final accuracy.
    """
    print('Training ' + model.__class__.__name__ + '\n')

    start_time = time()

    points = []

    for i in range(iterations):
        training_data = dataset.training()
        model.train(training_data[0], training_data[1])

        if print_every and i % print_every == 0:
            complete = i / iterations
            test_data = dataset.testing()
            accuracy = model.accuracy(test_data[0], test_data[1])

            points.append((i, float(accuracy)))

            print_args = (complete * 100, i, iterations, accuracy * 100, time() - start_time)

            print(
                'Training phase %.2f%% complete.\t(%d/%d)\tAccuracy:\t%6.2f%%\tElapsed:\t%.2fs'
                % print_args
            )

    end_avg = sum([x[1] for x in points[-5:]]) / 5
    print('\n' + '-' * 85)
    print(
        'Training complete.\t\tLast 5 Mean Accuracy:\t%6.2f%%\tElapsed %.2fs'
        % (end_avg * 100, time() - start_time)
    )

    print('\nConfusion over testing/training data')

    data = dataset.confusion()

    labels = list(data.keys())
    labels.sort(key=lambda x: dataset.m[x].argmax())

    print('\nact\\pred\t' + '\t'.join(labels))
    print('-' * 70)
    for label in labels:
        examples = np.vstack(data[label][1])
        counts = map(str, model.count_predicted_labels(examples))
        print('{}\t|\t{}'.format(label, '\t'.join(counts)))
        

    print('\nStarting validation phase')

    for vdata in dataset.validation():
        print(vdata[0])
        guess = model.count_predicted_labels(vdata[1])
        print(guess)

    return points


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

    for i in range(3, len(sys.argv)):
        if '=' in sys.argv[i]:
            key, value = sys.argv[i].split('=')
            kwargs[key] = __try_number(value)
        else:
            args.append(json.loads(sys.argv[i]))

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

    points = run(model, dataset)

    with open(sys.argv[1] + '_out.json', 'w') as out:
        json.dump(points, out)
