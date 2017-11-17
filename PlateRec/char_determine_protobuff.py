import tensorflow as tf
import os
import cv2
import numpy as np
from Graph import Graph

FREEZE_MODEL_PATH = os.path.abspath('./frozen_module/char-cnn')


class CharDetermine(object):
    def __init__(self):
        dir(tf.contrib)
        self.imgs = []
        self.in_imgs = []
        self.imgs_labels = []

    @staticmethod
    def __get_imgs(imgs):
        imgs_list = []

        for img in imgs:
            if img.shape[0] > 0 and img.shape[1] > 0:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                img = np.reshape(img, [-1, 784])
                imgs_list.append(img)

        return imgs_list

    @staticmethod
    def __load_graph(frozen_graph):
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='prefix')

        return graph

    def __char_detection(self):
        # graph = self.__load_graph(FREEZE_MODEL_PATH + \
        #                           '/frozen_model.pb')
        graph = Graph()

        # with tf.Session(graph=graph.graph_char) as sess:
        sess = graph.sess_char
        x = sess.graph.get_tensor_by_name('prefix/x:0')
        y = sess.graph.get_tensor_by_name('prefix/output/predict_sm:0')
        keep_prob = sess.graph.get_tensor_by_name('prefix/keep_prob:0')

        for i in range(len(self.in_imgs)):
            logits = sess.run(y, feed_dict={
                x: self.in_imgs[i],
                keep_prob: .5
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
