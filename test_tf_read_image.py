import tensorflow as tf


image_string = tf.read_file('demo.jpg')
print image_string
image_decoded = tf.image.decode_image(image_string)
dims = image_decoded.get_shape()
print (dims)

'''
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
'''
