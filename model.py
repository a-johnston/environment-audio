#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


def _one_hot(i, n):
    l = [0] * n
    l[i] = 1
    return l


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

       The dataset is passed as it provides the shapes of examples and labels,
       but model implementations shouldn't be able to see the dataset.
       
       When implementing a new model, overriding the build(...) method will
       allow you to construct your model with the variables x and y
       consistently defined.
    """
    session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16))

    x = None
    y = None

    __X__ = None
    __Y__ = None

    def __init__(self, dataset, *args, **kwargs):
        Model.x = tf.placeholder(tf.float32, shape=[None, dataset.x_shape()])
        Model.y = tf.placeholder(tf.float32, shape=[None, dataset.y_shape()])
        
        self.build(*args, **kwargs)

        if 'global_variables_initializer' in tf.__dir__():
            init = tf.global_variables_initializer()
        else:
            init = tf.initialize_all_variables()
        Model.session.run(init)

    @staticmethod
    def input_shape():
        return [Model.x.get_shape()[1].value]

    @staticmethod
    def output_shape():
        return [Model.y.get_shape()[1].value]

    @property
    def train_step(self):
        raise NotImplemented

    @property
    def classify(self):
        raise NotImplemented

    def build(self, dataset, *args, **kwargs):
        pass

    def train(self, X, Y):
        """Runs the training step of the model, if provided, with the given
           labeled examples.
        """
        if self.train_step:
            Model.__X__ = X
            Model.__Y__ = Y

            self.train_step.run(session=Model.session, feed_dict={Model.x: X, Model.y: Y})


    def count_predicted_labels(self, X):
        Model.__X__ = X

        argmax = tf.argmax(self.classify, 1).eval(session=Model.session, feed_dict={Model.x: X})
        return np.sum(np.vstack(map(_one_hot, argmax)), 0)


    def accuracy(self, X, Y):
        """Returns accuracy as percentage correctly predicted class labels
        """
        Model.__X__ = X
        Model.__Y__ = Y

        predicted = tf.equal(tf.argmax(self.classify, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

        return accuracy.eval(session=Model.session, feed_dict={Model.x: X, Model.y: Y})


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


def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,3,1,1], strides=[1, 3, 0, 1], padding='SAME')


class ConvNet(Model):
    def build(self, dataset, *args, **kwargs):
        W_conv1 = weight_variable()

    @property
    def train_step(self):
        pass

    @property
    def classify(self):
        pass


class SimpleFFNet(Model):
    _act_options = {
        'sigmoid': tf.sigmoid,
    }

    def build(self, hidden_layers=[], activation='sigmoid'):
        activation = SimpleFFNet._act_options[activation]

        last_layer = Model.x
        last_size = Model.input_shape()[0]

        for size in hidden_layers:
            W = weight_variable([last_size, size])
            b = bias_variable([size])

            W.initializer.run(session=Model.session)
            b.initializer.run(session=Model.session)

            last_layer = activation(tf.matmul(last_layer, W) + b)
            last_size = size

        W = weight_variable([last_size] + Model.output_shape())
        b = bias_variable(Model.output_shape())

        W.initializer.run(session=Model.session)
        b.initializer.run(session=Model.session)

        self.predicted_y = tf.matmul(last_layer, W) + b

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predicted_y, Model.y))
        # self._train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self._train_step = tf.train.AdamOptimizer(beta2=0.9).minimize(cross_entropy)

    @property
    def train_step(self):
        return self._train_step

    @property
    def classify(self):
        return self.predicted_y


class RandomClassifier(Model):
    @property
    def train_step(self):
        return None

    @property
    def classify(self):
        random = weight_variable(Model.__Y__.shape)
        random.initializer.run(session=Model.session)
        return random
