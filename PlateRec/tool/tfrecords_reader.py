import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class tfrecords_reader:
    def __init__(self, path):
        self.imgs = []
        self.labels = []
        self.tfrecord_path = path

    def main(self, batch):
        features = self._load_tfrecords()
        # imgs, labels = self._get_data_label(features, batch)
        # return self._read_data(imgs, labels)
        imgs, lbls = self._get_data_label(features, batch)
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            # init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            images, labels = sess.run([imgs, lbls])
            coord.request_stop()
            coord.join(threads)
        return images, labels

    def _load_tfrecords(self):
        tfrecords = os.listdir(self.tfrecord_path)
        # data_path = self.tfrecord_path + os.path.sep + tfrecords[0]
        data_path = self.tfrecord_path + os.path.sep + 'plates.tfrecords'
        filename_queue = tf.train.string_input_producer([data_path], num_epochs=1, name='queue')
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        feature = {'train/image': tf.FixedLenFeature([], tf.string),
                   'train/label': tf.FixedLenFeature([2], dtype=tf.int64)}
        features = tf.parse_single_example(serialized_example, features=feature, name='features')
        return features

    @staticmethod
    def _get_data_label(features, batch):
        image = tf.decode_raw(features['train/image'], tf.uint8)
        label = tf.cast(features['train/label'], tf.int32)
        image = tf.reshape(image, [20, 70, 3])
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch,
                                                num_threads=4,
                                                capacity=50000,
                                                min_after_dequeue=10000)
        # images, labels = tf.train.shuffle_batch([image, label],
        #                                        batch_size=batch,
        #                                        capacity=30,
        #                                        num_threads=1,
        #                                        min_after_dequeue=10)
        # images, labels = tf.train.batch([image, label],
        #                                 batch_size=batch,
        #                                 capacity=32,
        #                                 enqueue_many=False,
        #                                 num_threads=1)

        return images, labels

    # def _read_data(self, imgs, lbls):
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     with tf.Session(config=config) as sess:
    #         init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #         sess.run(init_op)
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(coord=coord)
    #         images, labels = sess.run([imgs, lbls])
    #         coord.request_stop()
    #         coord.join(threads)
    #     return images, labels


if __name__ == '__main__':
    path = os.path.abspath('../TFRecords')
    reader = tfrecords_reader(path)
    imgs, labels = reader.main(2267)
    print(imgs.shape,labels.shape)
    plt.imshow(imgs[2])
    plt.show()
