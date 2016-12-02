#!/usr/bin/env python3
import sys
from model import *
from dataset import *


def run(
    model,
    dataset,
    iterations=50000,
    print_every=1000,
):
    """Trains and evaluates the given model using the given dataset. Prints out
       progress and accuracy at a set interval as well as final accuracy.
    """
    for i in range(iterations):
        training_data = dataset.training()
        model.train(training_data[0], training_data[1])

        if print_every and i % print_every == 0:
            complete = i / iterations
            test_data = dataset.testing()
            accuracy = model.accuracy(test_data[0], test_data[1])

            print_args = (complete * 100, i, iterations, accuracy * 100)

            print(
                'Training phase %.2f%% complete.\t(%d/%d)\tAccuracy: %6.2f%%'
                % print_args
            )

    test_data = dataset.testing()
    accuracy = model.accuracy(test_data[0], test_data[1])
    print('')
    print('Training complete.\t\t\t\tAccuracy: %6.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No model specified.')
        print('Options: ' + ', '.join(Model.models.keys()))
        sys.exit(1)

    model = sys.argv[1]
    if model not in Model.models:
        print('Unknown model ' + model + '.')
        print('Options: ' + ', '.join(Model.models.keys()))
        sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2] == 'mock':
        dataset = Dataset.mock()
    else:
        dataset = Dataset.load_wavs()
    model = Model.models[model](dataset)

    run(model, dataset)
