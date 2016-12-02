#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class ModelMeta(type):
    """Metaclass for models used to index loaded models.
    """
    def __init__(cls, name, bases, dct):
        if not hasattr(cls, 'models'):
            cls.models = {}  # Happens for Model, not for subclasses
        else:
            cls.models[name] = cls
        super(ModelMeta, cls).__init__(name, bases, dct)


class Model(metaclass=ModelMeta):
    """Base model which can train and evaluate accuracy. Variables X and Y are
       vectors of examples, although ultimately this abstraction is handled by
       tensorflow.
    """

    def __init__(*args, **kwargs):
        pass

    def train(self, X, Y):
        raise NotImplemented

    def classify(self, X):
        raise NotImplemented
    
    def accuracy(self, X, Y):
        """Returns accuracy as percentage correctly predicted class labels
        """
        predicted = self.classify(X)
        comp = (predicted == Y).all(axis=1)
        num_correct = sum(map(lambda x: 1 if x else 0, comp))

        return num_correct / len(X)


def weight_variable(shape):
    """Creates a weight variable sampled from a normal distribution
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Creates a TF variables with a small positive intial bias
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class ConvNet(Model):
    def train(self, X, Y):
        pass

    def classify(self, x):
        pass


class FFNet(Model):
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
