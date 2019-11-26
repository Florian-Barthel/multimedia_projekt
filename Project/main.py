import tensorflow as tf
import data
import graph
import anchorgrid

f_map_rows = 20
f_map_cols = 20
scale_factor = 16.0
scales = [70, 100, 140, 200]
aspect_ratios = [0.5, 1.0, 2.0]
batch_size = 10
iou = 0.5

images_placeholder = tf.placeholder(tf.float32, shape=(None, None, None, 3))
labels_placeholder = tf.placeholder(tf.float32, shape=(None, f_map_rows, f_map_cols, len(scales), len(aspect_ratios), 1))

graph = graph.build(images_placeholder=images_placeholder,
                    labels_placeholder=labels_placeholder,
                    batch_size=batch_size,
                    num_scales=len(scales),
                    num_aspect_ratios=len(aspect_ratios))

my_anchor_grid = anchorgrid.anchor_grid(f_map_rows=f_map_rows,
                                        f_map_cols=f_map_cols,
                                        scale_factor=scale_factor,
                                        scales=scales,
                                        aspect_ratios=aspect_ratios)

batch_images, batch_labels = data.make_random_batch(batch_size, my_anchor_grid, iou)

with tf.compat.v1.Session() as sess:
    print('Session starting...')
    tf.compat.v1.global_variables_initializer().run()
    output = sess.run(graph, feed_dict={images_placeholder: batch_images,
                                        labels_placeholder: batch_labels})

    print(output)
