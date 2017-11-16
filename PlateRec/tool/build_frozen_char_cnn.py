import tensorflow as tf
import os

IMPORT_PATH = os.path.abspath('../module/char-cnn/2000_2')
EXPORT_PATH = os.path.abspath('../frozen_module/char-cnn')


def main():
    dir(tf.contrib)
    output_graph = EXPORT_PATH + '/frozen_model.pb'
    output_node_name = 'x,output/predict_sm,keep_prob'

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(IMPORT_PATH + '/char_classification_CNN.ckpt.meta')
        saver.restore(sess, IMPORT_PATH + '/char_classification_CNN.ckpt')

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_name.split(',')
        )

        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    main()
