from tqdm import tqdm
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import data
import graph
import anchorgrid

f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [70, 100, 140, 200]
aspect_ratios = [0.5, 0.75, 1.0, 1.5, 2.0]
batch_size = 20
iou = 0.5
learning_rate = 0.05
iterations = 1
negative_percentage = 0.05

images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                       320,
                                                       320,
                                                       3))

labels_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                       f_map_rows,
                                                       f_map_cols,
                                                       len(scales),
                                                       len(aspect_ratios),
                                                       1))

calculate_output = graph.output(images_placeholder=images_placeholder,
                                num_scales=len(scales),
                                num_aspect_ratios=len(aspect_ratios),
                                f_rows=f_map_rows,
                                f_cols=f_map_cols)

calculate_loss = graph.loss(input_tensor=calculate_output,
                            labels_placeholder=labels_placeholder,
                            negative_percentage=negative_percentage)

my_anchor_grid = anchorgrid.anchor_grid(f_map_rows=f_map_rows,
                                        f_map_cols=f_map_cols,
                                        scale_factor=scale_factor,
                                        scales=scales,
                                        aspect_ratios=aspect_ratios)


def optimize(my_loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    objective = optimizer.minimize(loss=my_loss)
    return objective


optimize = optimize(calculate_loss)

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    print('Session starting...')
    tf.compat.v1.global_variables_initializer().run()
    progress_bar = tqdm(range(iterations))
    for i in progress_bar:
        batch_images, batch_labels, _ = data.make_random_batch(batch_size=batch_size,
                                                               anchor_grid=my_anchor_grid,
                                                               iou=iou)

        loss, _ = sess.run([calculate_loss, optimize], feed_dict={images_placeholder: batch_images,
                                                                  labels_placeholder: batch_labels})

        description = ' loss:' + str(np.around(loss, decimals=5))
        progress_bar.set_description(description, refresh=True)

    num_test_images = 5
    test_images, test_labels, gt_annotation_rects = data.make_random_batch(num_test_images, my_anchor_grid, iou)
    output = sess.run(calculate_output, feed_dict={images_placeholder: test_images,
                                                   labels_placeholder: test_labels})

    for i in range(num_test_images):
        output_annotation_rects = data.convert_to_annotation_rects_output(my_anchor_grid, output[i])
        labels_annotation_rects = data.convert_to_annotation_rects_label(my_anchor_grid, test_labels[i])
        norm_img = ((test_images[i] + 1) * 128).astype(np.uint8)
        img = Image.fromarray(norm_img, 'RGB')
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=labels_annotation_rects,
                                 color=(255, 100, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=gt_annotation_rects[i],
                                 color=(100, 100, 255))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=output_annotation_rects,
                                 color=(100, 255, 100))
        img.save('test_images/max_overlap_boxes_{}.jpg'.format(i))
