#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import sys


if __name__ == '__main__':
    args = []
    base = 1472 / (1472 + 639)
    x = []

    colors = ['c', 'm', 'b']

    for f in sys.argv[1:]:
        with open(f) as data:
            points = json.load(data)
            c = colors.pop()
            max_accuracy = max([p[1] for p in points])

            x, y = zip(*points)

            args += [
                x, y, c,
                x, [max_accuracy] * len(x), c + '--',
            ]

    args += [
        x, [base] * len(x), 'r--',
    ]

    plt.plot(*args)

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.show()
