import numpy as np
import tensorflow as tf
import cv2
import os
import sys


class tfrecords_builder:
    def __init__(self):
        self.PLATES_ADDR = os.path.abspath('../../Plates/0')
        self.NON_PLATES_ADDR = os.path.abspath('../../Plates/1')
        self.IS_PLATE = 0
        self.NOT_PLATE = 1
        self.TFRECORDS_ADDR = os.path.abspath('../TFRecords')

    def _list_imgs_labels(self):
        imgs = []
        labels = []
        plates = os.listdir(self.PLATES_ADDR)
        for i in range(len(plates)):
            img = self._get_img(self.PLATES_ADDR + os.path.sep + str(plates[i]))
            imgs.append(img)
            labels.append(self.IS_PLATE)
        non_plates = os.listdir(self.NON_PLATES_ADDR)
        for i in range(len(non_plates)):
            img = self._get_img(self.NON_PLATES_ADDR + os.path.sep + str(non_plates[i]))
            imgs.append(img)
            labels.append(self.NOT_PLATE)
        return imgs, labels

    def _build_tfrecords(self, imgs, labels):
        file_name = self.TFRECORDS_ADDR + os.path.sep + 'plates.tfrecords'

        writer = tf.python_io.TFRecordWriter(file_name)

        for i in range(len(imgs)):
            img = imgs[i]
            label = labels[i]

            feature = {'train/label': self._int64_feature(label),
                       'train/image': self._bytes_feature(tf.compat.as_bytes(img.tostring()))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def main(self):
        imgs, labels = self._list_imgs_labels()
        self._build_tfrecords(imgs, labels)

    @staticmethod
    def _get_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    builder = tfrecords_builder()
    builder.main()
