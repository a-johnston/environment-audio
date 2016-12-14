#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import sys


if __name__ == '__main__':
    args = []
    x = []

    colors = ['g', 'c', 'm', 'b']

    for f in sys.argv[1:]:
        with open(f) as data:
            points = json.load(data)
            c = colors.pop()
            end_accuracy = sum([p[1] for p in points[-5:]])/5

            print(c, end_accuracy)

            x, y = zip(*points)

            args += [
                x, y, c,
                x, [end_accuracy] * len(x), c + '--',
            ]

    args += [
        x, [639 / (639 + 600 + 300 + 180 + 318 + 245)] * len(x), 'r--',
    ]

    plt.plot(*args)

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    plt.show()
