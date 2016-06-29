import tensorflow as tf
import matplotlib

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
#Hello, TensorFlow!
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a + b))