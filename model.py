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

       The dataset is passed as it provides the shapes of examples and labels
       as well as 
    """
    session = tf.Session()

    def __init__(dataset):
        pass

    @property
    def train_step(self):
        raise NotImplemented

    @property
    def classify(self):
        raise NotImplemented

    @property
    def _x(self):
        raise NotImplemented

    @property
    def _y(self):
        raise NotImplemented

    def train(self, X, Y):
        self.train_step.run(session=Model.session, feed_dict={self._x: X, self._y: Y})

    def accuracy(self, X, Y):
        """Returns accuracy as percentage correctly predicted class labels
        """
        predicted = tf.equal(tf.argmax(self.classify, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

        return accuracy.eval(session=Model.session, feed_dict={self._x: X, self._y: Y})


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
    @property
    def train_step(self):
        pass

    @property
    def classify(self):
        pass


class SimpleFFNet(Model):
    def __init__(self, dataset):
        x_shape = dataset.x_shape()
        y_shape = dataset.y_shape()

        self.x = tf.placeholder(tf.float32, shape=[None, x_shape])
        self.y = tf.placeholder(tf.float32, shape=[None, y_shape])

        W = weight_variable([x_shape, y_shape])
        b = bias_variable([y_shape])

        W.initializer.run(session=Model.session)
        b.initializer.run(session=Model.session)

        self.predicted_y = tf.matmul(self._x, W) + b

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predicted_y, self._y))
        self._train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    @property
    def train_step(self):
        return self._train_step

    @property
    def classify(self):
        return self.predicted_y

    @property
    def _x(self):
        return self.x

    @property
    def _y(self):
        return self.y

"""
class RandomClassifier(Model):
    def __init__(self, dataset):
        unique_labels = {tuple(row) for row in dataset.training()[1]}
        self.labels = np.vstack(unique_labels)

    @property
    def train_step(self):
        pass

    @property
    def classify(self):
        choices = np.random.randint(self.labels.shape[0], size=len(X))
        return self.labels[choices, :]
"""
