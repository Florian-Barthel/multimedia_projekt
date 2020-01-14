import tensorflow as tf
f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [80, 120, 150]
aspect_ratios = [1.0, 1.5, 2.0]
batch_size = 32
iou = 0.5
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.0001
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.97, staircase=True)
iterations = 40000
negative_example_factor = 10
