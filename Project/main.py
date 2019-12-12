from tqdm import tqdm
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import dataSet
import dataUtil
import data
import graph
import anchorgrid
import evaluation
import visualize

f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [50, 80, 100, 150]
aspect_ratios = [1.0, 1.5, 2.0]
batch_size = 30
iou = 0.5
learning_rate = 0.001
iterations = 10

negative_percentage = 15

visualize.run_tensorboard()

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

anchor_grid = anchorgrid.anchor_grid(f_map_rows=f_map_rows,
                                     f_map_cols=f_map_cols,
                                     scale_factor=scale_factor,
                                     scales=scales,
                                     aspect_ratios=aspect_ratios)

train_dataset = dataSet.create("./dataset_mmp/train", anchor_grid, iou).batch(batch_size)
test_dataset = dataSet.create("./dataset_mmp/test", anchor_grid, iou).batch(batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session(config=config) as sess:
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    images_tensor, labels_tensor = next_element

    #use only raw images!
    no_gts_images_tensor = images_tensor[:,0]

    calculate_output = graph.output(images=no_gts_images_tensor,
                                    num_scales=len(scales),
                                    num_aspect_ratios=len(aspect_ratios),
                                    f_rows=f_map_rows,
                                    f_cols=f_map_cols)

    calculate_loss, num_labels, num_random, num_weights, num_predicted = graph.loss(input_tensor=calculate_output,
                                                                                    labels=labels_tensor,
                                                                                    negative_percentage=negative_percentage)
    
    def optimize(my_loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        objective = optimizer.minimize(loss=my_loss)
        return objective


    optimize = optimize(calculate_loss)

    tf.summary.scalar('loss', calculate_loss)
    tf.summary.scalar('num_predicted', num_predicted)
    merged_summary = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(visualize.logs_directory, sess.graph, flush_secs=5)

    graph_vars = tf.global_variables()
    for var in tqdm(graph_vars):
        try:
            sess.run(var)
        except:
            print('found uninitialized variable {}'.format(var.name))
            sess.run(tf.initialize_variables([var]))

    progress_bar = tqdm(range(iterations))
    for i in progress_bar:
        batch_images, batch_labels, _, _ = data.make_random_batch(batch_size=batch_size,
                                                                  anchor_grid=anchor_grid,
                                                                  iou=iou)

        loss, labels, random, weights, predicted, _, summary = sess.run([calculate_loss, num_labels, num_random, num_weights, num_predicted, optimize, merged_summary], feed_dict={handle: train_handle})

        description = ' loss:' + str(np.around(loss, decimals=5)) + ' num_labels: ' + str(
            labels) + ' num_random: ' + str(random) + ' num_weights: ' + str(weights) + ' num_predicted: ' + str(
            predicted)
        progress_bar.set_description(description, refresh=True)
        # TensorBoard scalar summary
        log_writer.add_summary(summary, i)

    num_test_images = 50
    test_images, test_labels, gt_annotation_rects, test_paths = data.make_random_batch(num_test_images, anchor_grid,
                                                                                       iou)
    output = sess.run(calculate_output, feed_dict={images_placeholder: test_images,
                                                   labels_placeholder: test_labels})

    # Saving detections for evaluation purposes
    nms_boxes = evaluation.prepare_detections(output, anchor_grid, test_paths, num_test_images)
    # Drawing first 10 images before and after non-maximum-suppression
    visualize.draw_images(test_images, test_labels, output, anchor_grid, gt_annotation_rects, nms_boxes, num_test_images)

