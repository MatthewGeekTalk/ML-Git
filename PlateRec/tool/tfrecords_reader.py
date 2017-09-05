import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class tfrecords_reader:
    def __init__(self, path):
        self.imgs = []
        self.labels = []
        self.tfrecord_path = path

    def main(self):
        features = self._load_tfrecords()
        imgs, labels = self._get_data_label(features)
        return self._read_data(imgs, labels)

    def _load_tfrecords(self):
        tfrecords = os.listdir(self.tfrecord_path)
        data_path = os.path.abspath('../TFRecords') + os.path.sep + tfrecords[0]
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1, name='queue')
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([], tf.int64)}
        features = tf.parse_single_example(serialized_example, features=feature, name='features')
        return features

    @staticmethod
    def _get_data_label(features):
        image = tf.decode_raw(features['train/image'], tf.float32)
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [50, 180, 3])
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=10,
                                                capacity=30,
                                                num_threads=1,
                                                min_after_dequeue=10)

        return images, labels

    def _read_data(self, imgs, lbls):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, labels = sess.run([imgs, lbls])
            coord.request_stop()
            coord.join(threads)
        return images, labels


if __name__ == '__main__':
    path = os.path.abspath('../TFRecords')
    reader = tfrecords_reader(path)
    imgs, labels = reader.main()
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('1', img)
        cv2.waitKey(0)
