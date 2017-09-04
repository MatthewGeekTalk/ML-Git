import tensorflow as tf
import cv2
import os


class tfrecords_reader:
    def __init__(self, path):
        self.tfrecord_path = path

    def main(self):
        features = self._load_tfrecords()
        imgs, labels = self._get_data_label(features)
        return imgs, labels

    def _load_tfrecords(self):
        tfrecords = os.listdir(self.tfrecord_path)
        features = object
        for tfrecord in tfrecords:
            data_path = self.tfrecord_path + os.path.sep + tfrecord
            feature = {'train/image': tf.FixedLenFeature([], tf.string),
                       'train/label': tf.FixedLenFeature([], tf.int64)}
            filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example, features=feature)
            break
        return features

    @staticmethod
    def _get_data_label(features):
        image = tf.decode_raw(features['train/image'], tf.float32)
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [50, 180, 3])
        images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                                min_after_dequeue=10)
        return images, labels


if __name__ == '__main__':
    path = os.path.abspath('../TFRecords')
    reader = tfrecords_reader(path)
    imgs, labels = reader.main()
    # img = imgs[0, :]
    print(imgs.shape)
