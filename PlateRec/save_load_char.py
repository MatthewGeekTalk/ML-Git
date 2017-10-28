import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# #Prepare to feed input, i.e. feed_dict and placeholders
# w1 = tf.placeholder("float", name="w1")
# w2 = tf.placeholder("float", name="w2")
# b1= tf.Variable(2.0,name="bias")
# feed_dict ={w1:4,w2:8}
#
# #Define a test operation that we will restore
# w3 = tf.add(w1,w2)
# w4 = tf.multiply(w3,b1,name="op_to_restore")
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# #Create a saver object which will save all the variables
# saver = tf.train.Saver()
#
# #Run the operation by feeding input
# # print(sess.run(w4,feed_dict))
# #Prints 24 which is sum of (w1+w2)*b1
#
# #Now, save the graph
# saver.save(sess, save_path='./test',global_step=1000)

if __name__ == '__main__':
    img = cv2.imread('hu.jpg', 0)
    print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.reshape(img, [1, 784])
    # img = tf.reshape(img,[1,4200])
    sess = tf.Session()
    dir(tf.contrib)
    saver = tf.train.import_meta_graph('char_classification_CNN.ckpt.meta')
    saver.restore(sess, './char_classification_CNN.ckpt')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("output/predict_sm:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    feed_dict = {x: img, keep_prob: 0.5}
    print(sess.run(y, feed_dict=feed_dict))
    # print(sess.run("dense/weight:0"))
    # print(sess.run("dense/bias:0"))
    # print(img.shape)
    # print(x.shape)
