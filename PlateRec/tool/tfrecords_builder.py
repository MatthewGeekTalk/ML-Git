import numpy as np
import tensorflow as tf
import cv2
import os


class tfrecords_builder:
    def __init__(self):
        self.PLATES_ADDR = os.path.abspath('../../Plates/0')
        self.NON_PLATES_ADDR = os.path.abspath('../../Plates/1')
        self.IS_PLATE = 0
        self.NOT_PLATE = 1
        self.TFRECORDS_ADDR = os.path.abspath('../TFRecords')

    def _list_imgs_labels(self):
        imgs = []
        shapes = []
        labels = []
        plates = os.listdir(self.PLATES_ADDR)
        for i in range(len(plates)):
            img, shape = self._get_img(self.PLATES_ADDR + os.path.sep + str(plates[i]))
            imgs.append(img)
            shapes.append(shape)
            labels.append(self.IS_PLATE)
        non_plates = os.listdir(self.NON_PLATES_ADDR)
        for i in range(len(non_plates)):
            img, shape = self._get_img(self.NON_PLATES_ADDR + os.path.sep + str(non_plates[i]))
            imgs.append(img)
            shapes.append(shape)
            labels.append(self.NOT_PLATE)
        return imgs, shapes, labels

    def _build_tfrecords(self, imgs, shapes, labels):
        file_name = self.TFRECORDS_ADDR + os.path.sep + 'plates.tfrecords'

        writer = tf.python_io.TFRecordWriter(file_name)

        for i in range(len(imgs)):
            feature = {'train/label': self._int64_feature(labels[i]),
                       'train/shape': self._bytes_feature(shapes[i]),
                       'train/image': self._bytes_feature(imgs[i])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        writer.close()

    def main(self):
        imgs, shapes, labels = self._list_imgs_labels()
        self._build_tfrecords(imgs, shapes, labels)

    @staticmethod
    def _get_img(path):
        img = cv2.imread(path)
        img = np.asarray(img, np.uint8)
        shape = np.array(img.shape, np.int32)
        return img.tobytes(), shape.tobytes()

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    builder = tfrecords_builder()
    builder.main()
