import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath('./'))

from tfrecords_reader import tfrecords_reader


class deepcnn:
    def __init__(self, x):
        self.x = tf.reshape(x, [-1, 30, 180, 3])
        self.conv1_name = ""
        self.conv2_name = ""
        self.pool1_name = ""
        self.pool2_name = ""
        self.dense_name = ""
        self.output_name = ""
        self.dropout_name = ""
        self.conv1_weight_shape = []
        self.conv1_bias_shape = []
        self.conv2_weight_shape = []
        self.conv2_bias_shape = []
        self.dense_weight_shape = []
        self.dense_bias_shape = []
        self.output_weight_shape = []
        self.output_bias_shape = []

    def set_name(self, conv1, conv2, pool1, pool2, dense, output, dropout):
        self.conv1_name = conv1
        self.conv2_name = conv2
        self.pool1_name = pool1
        self.pool2_name = pool2
        self.dense_name = dense
        self.output_name = output
        self.dropout_name = dropout

    def set_conv1_shape(self, weight_shape, bias_shape):
        self.conv1_weight_shape = weight_shape
        self.conv1_bias_shape = bias_shape

    def set_conv2_shape(self, weight_shape, bias_shape):
        self.conv2_weight_shape = weight_shape
        self.conv2_bias_shape = bias_shape

    def set_dense_shape(self, weight_shape, bias_shape):
        self.dense_weight_shape = weight_shape
        self.dense_bias_shape = bias_shape

    def set_output_shape(self, weight_shape, bias_shape):
        self.output_weight_shape = weight_shape
        self.output_bias_shape = bias_shape

    def get_conv1_shape(self):
        return self.conv1_weight_shape, self.conv1_bias_shape

    def get_conv2_shape(self):
        return self.conv2_weight_shape, self.conv2_bias_shape

    def get_dense_shape(self):
        return self.dense_weight_shape, self.dense_bias_shape

    def get_output_shape(self):
        return self.output_weight_shape, self.output_bias_shape

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def _build_dropout(name, x):
        with tf.name_scope(name=name):
            keep_prob = tf.placeholder(tf.float32)
            h_dense_drop = tf.nn.dropout(x, keep_prob)
            return h_dense_drop

    def _build_conv(self, name, weight_shape, bias_shape, x):
        with tf.name_scope(name=name):
            W_conv1 = self._weight_variable(weight_shape)
            b_conv1 = self._bias_variable(bias_shape)
            h_conv1 = tf.nn.relu(self._conv2d(x, W_conv1) + b_conv1)
            return h_conv1

    def _build_pool(self, name, conv):
        with tf.name_scope(name=name):
            h_pool = self._max_pool_2x2(conv)
            return h_pool

    def _build_dense(self, name, weight_shape, bias_shape, x):
        with tf.name_scope(name=name):
            W_dense = self._weight_variable(weight_shape)
            b_dense = self._bias_variable(bias_shape)
            h_pool2_flat = tf.reshape(x, [-1, weight_shape[0]])
            h_dense = tf.nn.relu(tf.matmul(h_pool2_flat, W_dense) + b_dense)
            return h_dense

    def _build_output(self, name, weight_shape, bias_shape, x):
        with tf.name_scope(name=name):
            W_output = self._weight_variable(weight_shape)
            b_output = self._bias_variable(bias_shape)

            y_conv = tf.matmul(x, W_output) + b_output
            return y_conv

    def build_cnn(self):
        h_conv1 = self._build_conv(self.conv1_name, self.conv1_weight_shape, self.conv1_bias_shape, self.x)
        h_pool1 = self._build_pool(self.pool1_name, h_conv1)
        h_conv2 = self._build_conv(self.conv2_name, self.conv2_weight_shape, self.conv2_bias_shape, h_pool1)
        h_pool2 = self._build_pool(self.pool2_name, h_conv2)
        h_dense = self._build_dense(self.dense_name, self.dense_weight_shape, self.dense_bias_shape, h_pool2)
        h_dense_drop = self._build_dropout(self.dropout_name, h_dense)
        y_conv = self._build_dense(self.output_name, self.output_weight_shape, self.output_bias_shape, h_dense_drop)
        return y_conv


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 16200])
    y_ = tf.placeholder(tf.float32, [None, 2])

    cnn = deepcnn()

    cnn.set_name(conv1='conv1', conv2='conv2', pool1='max pool1', pool2='max pool2', dense='full connection',
                 output='final output', dropout='drop out')
    # cnn.set_conv1_shape([5, 5, 3, 32], [32])
    # cnn.set_conv2_shape()
