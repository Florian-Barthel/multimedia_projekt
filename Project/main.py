from tqdm import tqdm
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from datetime import datetime
import data
import graph
import anchorgrid
import evaluation
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [70, 100, 140, 200]
aspect_ratios = [0.5, 1.0, 2.0]
batch_size = 32
iou = 0.5
learning_rate = 0.001
iterations = 1000
negative_percentage = 10

# TensorBoard logs saved in ./logs/dd-MM-yyyy_HH-mm-ss
current_time = datetime.now()
logs_directory = './logs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S')

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 320, 320, 3))

    labels_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                           f_map_rows,
                                                           f_map_cols,
                                                           len(scales),
                                                           len(aspect_ratios)))

    calculate_output = graph.output(images_placeholder=images_placeholder,
                                    num_scales=len(scales),
                                    num_aspect_ratios=len(aspect_ratios),
                                    f_rows=f_map_rows,
                                    f_cols=f_map_cols)

    calculate_loss, num_labels, num_random, num_weights, num_predicted = graph.loss(input_tensor=calculate_output,
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

    graph_vars = tf.global_variables()
    for var in tqdm(graph_vars):
        try:
            sess.run(var)
        except:
            print('found uninitialized variable {}'.format(var.name))
            sess.run(tf.initialize_variables([var]))

    # TensorBoard graph summary
    tf.summary.scalar('loss', calculate_loss)
    tf.summary.scalar('num_predicted', num_predicted)
    merged_summary = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(logs_directory, sess.graph)
    progress_bar = tqdm(range(iterations))
    for i in progress_bar:
        batch_images, batch_labels, _, test_paths = data.make_random_batch(batch_size=batch_size,
                                                                           anchor_grid=my_anchor_grid,
                                                                           iou=iou)

        loss, labels, random, weights, predicted, _, summary = sess.run(
            [calculate_loss, num_labels, num_random, num_weights, num_predicted, optimize, merged_summary],
            feed_dict={images_placeholder: batch_images,
                       labels_placeholder: batch_labels})

        description = ' loss:' + str(np.around(loss, decimals=5)) + ' num_labels: ' + str(
            labels) + ' num_random: ' + str(random) + ' num_weights: ' + str(weights) + ' num_predicted: ' + str(
            predicted)
        progress_bar.set_description(description, refresh=True)
        # TensorBoard scalar summary
        log_writer.add_summary(summary, i)

    validation_data = data.get_validation_data(100, my_anchor_grid, iou)
    # test_images, test_labels, gt_annotation_rects, test_paths = data.get_validation_data(my_anchor_grid, iou)
    for i in range(len(validation_data)):
        (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[i]
        output = sess.run(calculate_output, feed_dict={images_placeholder: test_images,
                                                       labels_placeholder: test_labels})
        # Saving detections for evaluation purposes
        evaluation.prepare_detections(output, my_anchor_grid, test_paths)

    num_view_images = 5
    for i in range(num_view_images):
        img = Image.fromarray(((test_images[i] + 1) * 128).astype(np.uint8), 'RGB')
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=data.convert_to_annotation_rects_label(my_anchor_grid, test_labels[i]),
                                 color=(255, 100, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=data.convert_to_annotation_rects_output(my_anchor_grid, output[i]),
                                 color=(100, 255, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=gt_annotation_rects[i],
                                 color=(100, 100, 255))
        if not os.path.exists('test_images'):
            os.makedirs('test_images')
        img.save('test_images/max_overlap_boxes_{}.jpg'.format(i))
