#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
import sys


if __name__ == '__main__':
    for f in sys.argv[1:]:
        with open(f) as data:
            points = json.load(data)
            plt.plot(*zip(*points))
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
    plt.show()
