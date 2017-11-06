import tensorflow as tf
import os

import_path = os.path.abspath('../module/bc-cnn2')
export_path = os.path.abspath('../frozen_module/bc-cnn2')


def main():
    dir(tf.contrib)
    graph = tf.get_default_graph()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(import_path + '/binary_classification_CNN.ckpt.meta')
        saver.restore(sess, import_path + '/binary_classification_CNN.ckpt')

        tf.train.write_graph(sess.graph_def, export_path, "binary_classification_CNN.pb", False)
        tf.train.write_graph(sess.graph_def, export_path, "binary_classification_CNN.pbtxt", True)


if __name__ == '__main__':
    main()
