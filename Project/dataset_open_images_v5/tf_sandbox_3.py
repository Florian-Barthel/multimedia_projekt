import tensorflow as tf

adjustment = tf.constant([[[[1, 2, 3, 4], [4, 5, 5, 4]], [[1, 2, 3, 3], [4, 5, 5, 4]]],
                          [[[6, 2, 3, 4], [4, 5, 5, 4]], [[10, 3, 3, 12], [1, 5, 5, 4]]],
                          [[[6, 2, 3, 4], [4, 5, 5, 4]], [[10, 3, 3, 12], [1, 5, 5, 4]]]])

with tf.Session() as sess:
    a = sess.run([adjustment])
    print('a')


