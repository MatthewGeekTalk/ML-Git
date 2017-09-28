import numpy as np
import tensorflow as tf
import sys
import os
import tempfile

sys.path.append(os.path.abspath('./tool/'))
from tfrecords_reader import tfrecords_reader

BATCH_SIZE = 50
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONUNBUFFERED"] = "0"


class deepcnn(object):
    def __init__(self, x):
        self.x = tf.reshape(x, [-1, 20, 70, 3])
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

        self.keep_prob = 0

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

    def set_keep_prob(self, keep_prob):
        self.keep_prob = keep_prob

    def get_conv1_shape(self):
        return self.conv1_weight_shape, self.conv1_bias_shape

    def get_conv2_shape(self):
        return self.conv2_weight_shape, self.conv2_bias_shape

    def get_dense_shape(self):
        return self.dense_weight_shape, self.dense_bias_shape

    def get_output_shape(self):
        return self.output_weight_shape, self.output_bias_shape

    def get_keep_prob(self):
        return self.keep_prob

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, dtype=tf.float32, name='weight')

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, dtype=tf.float32, name='bias')

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name='conv2d')

    @staticmethod
    def _max_pool(x, width, height):
        return tf.nn.max_pool(x, ksize=[1, width, height, 1],
                              strides=[1, width, height, 1], padding='SAME',name='max_pool')

    @staticmethod
    def _build_dropout(name, x, keep_prob):
        with tf.name_scope(name=name):
            h_dense_drop = tf.nn.dropout(x, keep_prob,name='dense_drop')
            return h_dense_drop

    def _build_conv(self, name, weight_shape, bias_shape, x):
        with tf.name_scope(name=name):
            W_conv1 = self._weight_variable(weight_shape)
            b_conv1 = self._bias_variable(bias_shape)
            h_conv1 = tf.nn.relu(self._conv2d(x, W_conv1) + b_conv1,name='conv_relu')
            return h_conv1

    def _build_pool(self, name, conv, width, height):
        with tf.name_scope(name=name):
            h_pool = self._max_pool(conv, width=width, height=height)
            return h_pool

    def _build_dense(self, name, weight_shape, bias_shape, x):
        with tf.name_scope(name=name):
            W_dense = self._weight_variable(weight_shape)
            b_dense = self._bias_variable(bias_shape)
            h_pool2_flat = tf.reshape(x, [-1, weight_shape[0]],name='pool2_flat')
            h_dense = tf.nn.relu(tf.matmul(h_pool2_flat, W_dense) + b_dense, name='dense_relu')
            return h_dense

    def _build_output(self, name, weight_shape, bias_shape, x):
        with tf.name_scope(name=name):
            W_output = self._weight_variable(weight_shape)
            b_output = self._bias_variable(bias_shape)

            y_conv = tf.add(tf.matmul(x, W_output), b_output, name='predict')
            y_conv_sm = tf.nn.softmax(y_conv, name='predict_sm')
            return y_conv

    def build_cnn(self):
        h_conv1 = self._build_conv(self.conv1_name, self.conv1_weight_shape, self.conv1_bias_shape, self.x)
        h_pool1 = self._build_pool(self.pool1_name, h_conv1, 2, 2)
        h_conv2 = self._build_conv(self.conv2_name, self.conv2_weight_shape, self.conv2_bias_shape, h_pool1)
        h_pool2 = self._build_pool(self.pool2_name, h_conv2, 2, 2)
        h_dense = self._build_dense(self.dense_name, self.dense_weight_shape, self.dense_bias_shape, h_pool2)
        h_dense_drop = self._build_dropout(self.dropout_name, h_dense, self.keep_prob)
        y_conv = self._build_output(self.output_name, self.output_weight_shape, self.output_bias_shape, h_dense_drop)
        return y_conv


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 20 * 70 * 3], name='x')
    y_ = tf.placeholder(tf.float32, [None, 1], name='y_')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    cnn = deepcnn(x)

    cnn.set_name(conv1='conv1', conv2='conv2', pool1='pool1', pool2='pool2', dense='dense',
                 output='output', dropout='dropout')
    cnn.set_conv1_shape([5, 5, 3, 32], [32])
    cnn.set_conv2_shape([5, 5, 32, 64], [64])
    cnn.set_dense_shape([5 * 18 * 64, 1024], [1024])
    cnn.set_output_shape([1024, 1], [1])
    cnn.set_keep_prob(keep_prob)
    y_conv = cnn.build_cnn()

    with tf.name_scope(name='loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv, name="cross_entropy")

    cross_entropy = tf.reduce_mean(cross_entropy, name="reduce_ce")

    with tf.name_scope(name='adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction, name='accuracy_1')

    # graph_location = tempfile.mkdtemp()
    # print('Saving  graph to: %s' % graph_location)
    # train_writer = tf.summary.FileWriter(graph_location)
    # train_writer.add_graph(tf.get_default_graph())

    # path = os.path.abspath('./TFRecords')
    path = os.path.abspath('/nfs/users/matthew/workdir')
    reader = tfrecords_reader(path)

    # MODEL_PATH = os.path.abspath('./net_structure/binary_classification_CNN.ckpt')
    # path at sap gpu server
    MODEL_PATH = '/nfs/users/matthew/saved_model/binary_classification_CNN.ckpt'
    saver = tf.train.Saver()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
    with tf.Session() as sess:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        # init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(36):
            imgs, labels = reader.main(batch=BATCH_SIZE)
            imgs = np.reshape(imgs, [BATCH_SIZE, 20 * 70 * 3])
            labels = np.reshape(labels, [BATCH_SIZE, 1])
            if i % 2 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: imgs, y_: labels, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: imgs, y_: labels, keep_prob: 0.5})
        writer = tf.summary.FileWriter("/nfs/users/matthew/saved_model", sess.graph)
        writer.close()
        save_path = saver.save(sess, MODEL_PATH)
        print('Model save at %s' % save_path)
