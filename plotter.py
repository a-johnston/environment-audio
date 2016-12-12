#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import sys


if __name__ == '__main__':
    for f in sys.argv[1:]:
        with open(f) as data:
            points = json.load(data)
            max_accuracy = max([p[1] for p in points])
            base = 1472 / (1472 + 639)

            x, y = zip(*points)

            plt.plot(
                x, y, 'b',
                x, [max_accuracy] * len(x), 'g--',
                x, [base] * len(x), 'r--',
            )
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
    plt.show()
