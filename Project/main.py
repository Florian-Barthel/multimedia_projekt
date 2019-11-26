import tensorflow as tf
from PIL import Image
import data
import graph
import numpy as np

scales = [70, 100, 140, 200]
aspect_ratios = [0.5, 1.0, 2.0]

image_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))

pre_trained_graph = graph.mobile_net_v2()

norm = graph.normalize(image_placeholder)
net = pre_trained_graph(norm)
conv = graph.convolution(net, len(scales), len(aspect_ratios))

train_data = data.get_dict_from_folder('train')
item = train_data.popitem()
path = item[0]
img = Image.open(path)
x = np.array(img, dtype=np.float32)
x = np.expand_dims(x, axis=0)
# x = np.expand_dims(x, axis=-1)

# output_placeholder = tf.placeholder(tf.float32, shape=(None, 1000))

with tf.compat.v1.Session() as sess:
    print('Session starting...')
    tf.compat.v1.global_variables_initializer().run()
    output = sess.run(conv, feed_dict={image_placeholder: x})
    print(output)
