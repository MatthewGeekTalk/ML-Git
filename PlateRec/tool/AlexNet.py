# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
import tensorflow as tf

NUM_CLASSES = 1000
IMAGE_SIZE = 224
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_dir', 'log', 'Directory to save tensorboard logs')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate.')


def inference(images_placeholder, keep_prob):
    def weight_variable(shape, num):
        initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(num)))
        return (tf.Variable(initial).initialized_value())

    def bias_variable(shape):
        initial = tf.zeros(shape)
        return (tf.Variable(initial).initialized_value())

    def conv2d(x, W):
        return (tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

    def max_pool_3x3(x):
        return (tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME'))

    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([11, 11, 3, 96], IMAGE_SIZE * IMAGE_SIZE)
        b_conv1 = bias_variable([96])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_3x3(tf.nn.local_response_normalization(h_conv1))

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 96, 256], 96)
        b_conv2 = bias_variable([256])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_3x3(tf.nn.local_response_normalization(h_conv2))

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 256, 384], 256)
        b_conv3 = bias_variable([384])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([3, 3, 384, 384], 384)
        b_conv4 = bias_variable([384])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_variable([3, 3, 384, 256], 384)
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_3x3(h_conv5)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 256, 4096], (7 * 7 * 256))
        b_fc1 = bias_variable([4096])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([4096, 4096], 4096)
        b_fc2 = bias_variable([4096])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([4096, NUM_CLASSES], 4096)
        b_fc3 = bias_variable([NUM_CLASSES])

        y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    return (y_conv)


def loss(logits, labels):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(tf.nn.softmax(logits))))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return (cross_entropy)


def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return (train_step)


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)
    return (accuracy)


if __name__ == '__main__':
    train_image = []
    train_label = []

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder('float32', shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder('float32', shape=(None, NUM_CLASSES))
        keep_prob = tf.placeholder('float')

        logits = inference(images_placeholder, keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, FLAGS.learning_rate)
        acc = accuracy(logits, labels_placeholder)

        save_path = 'models/'
        model_name = 'model1.ckpt'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver = tf.train.Saver()
        save_path_full = os.path.join(save_path, model_name)

        with tf.Session() as sess:
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            feed_dict_not_dropout = {
                images_placeholder: train_image[:],
                labels_placeholder: train_label[:],
                keep_prob: 1.0}
            batch_step = int(len(train_image) / FLAGS.batch_size)

            for step in range(FLAGS.max_steps):
                for i in range(batch_step):
                    batch = FLAGS.batch_size * i
                    sess.run(train_op, feed_dict={
                        images_placeholder: train_image[batch:batch + FLAGS.batch_size],
                        labels_placeholder: train_label[batch:batch + FLAGS.batch_size],
                        keep_prob: 0.5})

                train_accuracy, summary_str = sess.run([acc, summary_op], feed_dict_not_dropout)
                summary_writer.add_summary(summary_str, step)

                print("step %d, training accuracy: %g" % (step, train_accuracy))

            saver.save(sess, save_path_full)
