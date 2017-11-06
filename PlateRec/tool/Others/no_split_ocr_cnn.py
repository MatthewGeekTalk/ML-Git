import tensorflow as tf
import numpy as np
import os
import math


def ocr_cnn(images, keep_prob):
    parameters = []

    # 36 * 180 * 3
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 3, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.Variable(tf.constant(.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        conv1 = tf.nn.bias_add(conv, biases)
        parameters += [kernel, biases]

    # 18 * 90 * 64
    with tf.name_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # 8 * 44 * 64
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name='relu')
        parameters += [kernel, biases]

    # 8 * 44 * 192
    with tf.name_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # 3 * 21 * 192
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(.0, shape=[384]), dtype='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name='relu')
        parameters += [kernel, biases]

    # 3 * 21 *384
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name='relu')
        parameters += [kernel, biases]

    # 3 * 21 * 256
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name='relu')
        parameters += [kernel, biases]

    # 3 * 21 * 256
    with tf.name_scope('pool3') as scope:
        pool3 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool3')

    # 2 * 10 * 256
    with tf.name_scope('fc1') as scope:
        Weights = tf.Variable(tf.truncated_normal([2 * 10 * 256, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        h_pool3_flat = tf.reshape(pool3, [-1, 2 * 10 * 256], name='pool3_flat')
        fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('dropout1') as scope:
        dropout1 = tf.nn.dropout(fc1, keep_prob=keep_prob, name='fc1_dropout')

    # 2 * 10 * 256
    with tf.name_scope('fc21') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc21 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('fc22') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc22 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('fc23') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc23 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('fc24') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc24 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('fc25') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc25 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('fc26') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc26 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    # 2 * 10 * 256
    with tf.name_scope('fc27') as scope:
        Weights = tf.Variable(tf.truncated_normal([67, 67], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(.0, shape=[67], dtype=tf.float32), trainable=True, name='biases')
        fc27 = tf.nn.relu(tf.matmul(dropout1, Weights) + biases, name='relu')

    with tf.name_scope('output') as scope:
        output = tf.concat([fc21, fc22, fc23, fc24, fc25, fc26, fc27], axis=0)

    return output, parameters


if __name__ == '__main__':
    pass
