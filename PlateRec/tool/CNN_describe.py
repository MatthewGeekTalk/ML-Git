import matplotlib.image as mpimg
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

if __name__ == '__main__':
    dir(tf.contrib)
    img = mpimg.imread(os.path.abspath('../0.jpg'))
    img = cv2.resize(img, (70, 20), cv2.INTER_CUBIC)
    img = np.reshape(img, (-1, 4200))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../../Test/bc-cnn2/binary_classification_CNN.ckpt.meta')
        saver.restore(sess, '../../Test/bc-cnn2/binary_classification_CNN.ckpt')
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('x:0')
        conv1 = graph.get_tensor_by_name('conv1/conv_relu:0')
        conv2 = graph.get_tensor_by_name('conv2/conv_relu:0')

        feature_1 = sess.run(conv1, feed_dict={
            x: img
        })

        feature_2 = sess.run(conv2, feed_dict={
            x: img
        })

    feature1 = feature_1[0, :]
    feature2 = feature_2[0, :]

    for i in range(feature1.shape[2]):
        img = feature1[:, :, i]
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        print('Conv 1 %i' % i)

    print(feature2.shape)

    for i in range(feature2.shape[2]):
        img = feature2[:, :, i]
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        print('Conv 2 %i' % i)
