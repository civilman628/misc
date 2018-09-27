import tensorflow as tf
#import tensorflow.contrib.slim as slim
from nets import inception_v2 #, inception_v1_arg_scope


x = tf.placeholder(shape=[128,224,224,224], dtype=tf.float32)
inception_v2 = inception_v2.inception_v2(x)

sess = tf.Session()
tf.train.write_graph(sess.graph_def, './', 'train.pbtxt')