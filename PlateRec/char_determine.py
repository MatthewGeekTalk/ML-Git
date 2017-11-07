import os
import numpy as np
import tensorflow as tf
import cv2

MODEL_PATH = os.path.abspath('./module/char-cnn/2000')
GRAPH = 'char_classification_CNN.ckpt.meta'
SESS = 'char_classification_CNN.ckpt'


class CharDetermine(object):
    def __init__(self):
        dir(tf.contrib)
        self.imgs = []
        self.in_imgs = []
        self.imgs_labels = []

        self.Graph = tf.Graph()  # initialize new graph for multiple model loading

    @staticmethod
    def __get_imgs(imgs):
        imgs_list = []

        for img in imgs:
            if img.shape[0] > 0 and img.shape[1] > 0:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                img = np.reshape(img, [-1, 784])
                imgs_list.append(img)
        return imgs_list

    def __char_detection(self):
        with tf.Session(graph=self.Graph) as sess:
            init = tf.global_variables_initializer()
            saver = tf.train.import_meta_graph(MODEL_PATH + os.sep + GRAPH)
            graph = tf.get_default_graph()
            saver.restore(sess, MODEL_PATH + os.sep + SESS)
            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('output/predict_sm:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')

            sess.run(init)

            for i in range(len(self.in_imgs)):
                logits = sess.run(y, feed_dict={
                    x: self.in_imgs[i], keep_prob: .5
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
