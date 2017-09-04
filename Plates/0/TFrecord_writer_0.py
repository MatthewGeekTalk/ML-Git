import tensorflow as tf
import numpy as np
import cv2
import sys
from random import shuffle
import glob

shuffle_data = True
rose_train_path = '*.jpg'
addrs = glob.glob(rose_train_path)

labels = []
for i in range(len(addrs)):
    labels.append(0) #0 is plate

if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

print(len(addrs))
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(labels)):]
train_addrs = addrs
train_labels = labels

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


train_filename = 'train.tfrecords_0'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_addrs)):
    # Load the image
    img = load_image(train_addrs[i])
    label = train_labels[i]
    # Create a feature
    feature = {'train/label_0': _int64_feature(label),
               'train/image_0': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()