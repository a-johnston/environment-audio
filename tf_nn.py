#!/usr/bin/env python3
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def create_model(shape):
    x = tf.placeholder(tf.float32, shape=[None, shape[0]])
    y = tf.placeholder(tf.float32, shape=[None, shape[-1]])
