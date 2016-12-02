#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Model:
    """Base model which can train and evaluate accuracy. Variables X and Y are
       vectors of examples, although ultimately this abstraction is handled by
       tensorflow.
    """
    def train(self, X, Y):
        raise NotImplemented

    def classify(self, X):
        raise NotImplemented
    
    def accuracy(self, X, Y):
        predicted = self.classify(X)
        comp = (predicted == Y).all(axis=1)
        num_correct = sum(map(lambda x: 1 if x else 0, comp))

        return num_correct / len(X)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class ConvNet(Model):
    def train(self, X, Y):
        pass

    def classify(self, x):
        pass


class RandomClassifier(Model):
    def __init__(self, dataset):
        self.session = tf.Session()
        self.labels = np.vstack({tuple(row) for row in dataset.training()[1]})

    def train(self, X, Y):
        pass

    def classify(self, X):
        return self.labels[np.random.randint(self.labels.shape[0], size=len(X)), :]
