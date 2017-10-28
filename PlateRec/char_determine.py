import os
import numpy as np
import tensorflow as tf
import cv2

MODEL_PATH = os.path.abspath('./module/char-cnn')
GRAPH = 'char_classification_CNN.ckpt.meta'
SESS = 'char_classification_CNN.ckpt'


class CharDetermine(object):
    def __init__(self):
        dir(tf.contrib)
        self.imgs = []
        self.in_imgs = []
        self.imgs_labels = []

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.import_meta_graph(MODEL_PATH + os.sep + GRAPH)
        self.graph = tf.get_default_graph()
        self.x = self.graph.get_tensor_by_name('x:0')
        self.y = self.graph.get_tensor_by_name('output/predict_sm:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')

    @staticmethod
    def __get_imgs(imgs):
        imgs_list = []

        for img in imgs:
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            img = np.reshape(img, [-1, 784])
            imgs_list.append(img)
        return imgs_list

    def __char_detection(self):
        with tf.Session() as sess:
            sess.run(self.init)
            self.saver.restore(MODEL_PATH + os.sep + SESS)
            for i in range(len(self.in_imgs)):
                logits = sess.run(self.y, feed_dict={
                    self.x: self.in_imgs[i], self.keep_prob: .5
                })

                logits = np.reshape(logits, [45])
                logits = np.asarray(logits, dtype=np.int32)
                self.imgs_labels.append(list(logits))

    def main(self, imgs):
        self.imgs = imgs
        self.in_imgs = self.__get_imgs(imgs=imgs)
        self.__char_detection()
        return self.imgs, self.imgs_labels


if __name__ == '__main__':
    pass
