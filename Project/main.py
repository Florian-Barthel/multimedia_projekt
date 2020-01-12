from tqdm import tqdm
import numpy as np
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import dataSet
import dataUtil
from datetime import datetime
import graph
import anchorgrid
import evaluation
from tensorboard import program
import mobilenet

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [50, 80, 100, 150]
aspect_ratios = [0.5, 0.75, 1.0, 1.5]
batch_size = 10
iou = 0.5
learning_rate = 0.0005
iterations = 150

negative_example_factor = 10

# TensorBoard logs saved in ./logs/dd-MM-yyyy_HH-mm-ss
current_time = datetime.now()
logs_directory = './logs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S')

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)

anchor_grid = anchorgrid.anchor_grid(f_map_rows=f_map_rows,
                                     f_map_cols=f_map_cols,
                                     scale_factor=scale_factor,
                                     scales=scales,
                                     aspect_ratios=aspect_ratios)

anchor_grid_tensor = tf.constant(anchor_grid, dtype=tf.int32)

train_dataset = dataSet.create("./dataset_mmp/train", batch_size)
test_dataset = dataSet.create("./dataset_mmp/test", batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()
test_iterator = test_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session(config=config) as sess:
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    images_tensor, gt_labels_tensor = next_element

    labels_tensor = dataUtil.calculate_overlap_boxes_tensor(gt_labels_tensor, anchor_grid, iou)

    features = mobilenet.mobile_net_v2()(images_tensor)

    probabilities = graph.probabilities_output(features, anchor_grid)
    probabilities_loss, num_labels, num_weights, num_predicted = graph.probabilities_loss(probabilities, labels_tensor, negative_example_factor)

    adjustments = graph.adjustments_output(features, anchor_grid, anchor_grid_tensor)
    adjustments_loss = graph.adjustments_loss(adjustments, gt_labels_tensor, labels_tensor, anchor_grid_tensor)

    total_loss = probabilities_loss * adjustments_loss

    def optimize(target_loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        objective = optimizer.minimize(target_loss)
        return objective

    optimize = optimize(total_loss)
    

    tf.summary.scalar('probabilities/Loss', probabilities_loss)
    tf.summary.scalar('probabilities/Number of positivies', num_predicted)
    tf.summary.scalar('adjustments/Loss', adjustments_loss)
    merged_summary = tf.summary.merge_all()

    graph_vars = tf.global_variables()
    for var in tqdm(graph_vars):
        try:
            sess.run(var)
        except:
            print('found uninitialized variable {}'.format(var.name))
            sess.run(tf.initialize_variables([var]))

    # TensorBoard graph summary
    log_writer = tf.summary.FileWriter(logs_directory, sess.graph, flush_secs=1)
    progress_bar = tqdm(range(iterations))
    for i in progress_bar:

        loss, labels, weights, predicted, _, summary, = sess.run([[total_loss, probabilities_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize, merged_summary], feed_dict={handle: train_handle})

        description = ' loss:' + str(np.around(loss, decimals=5)) + ' num_labels: ' + str(
            labels) + ' num_weights: ' + str(weights) + ' num_predicted: ' + str(
            predicted)
        progress_bar.set_description(description, refresh=True)
        # # TensorBoard scalar summary
        log_writer.add_summary(summary, i)


    gt_images_tensor = tf.image.draw_bounding_boxes(images_tensor, gt_labels_tensor)
    ag_adjusted_tensor = dataUtil.calculate_adjusted_anchor_grid(anchor_grid_tensor, adjustments)

    images_result, labels_result, probabilities_predicted, adjustments_predicted, ag_adjusted = sess.run([gt_images_tensor, labels_tensor, probabilities, adjustments, ag_adjusted_tensor], feed_dict={handle: test_handle})

    output_image_size = (720, 720)    

    test_paths = []
    for i in range(np.shape(images_result)[0]):
        image = Image.fromarray(((images_result[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        image.resize(output_image_size, Image.ANTIALIAS).save('test_images/{}_gts.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_label(anchor_grid, labels_result[i]), (0, 255, 255))
        image.resize(output_image_size, Image.ANTIALIAS).save('test_images/{}_labels.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_output(anchor_grid, probabilities_predicted[i]), (0, 0, 255))
        image.resize(output_image_size, Image.ANTIALIAS).save('test_images/{}_estimates.jpg'.format(i))


        image_adjusted = Image.fromarray(((images_result[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        
        dataUtil.draw_bounding_boxes(image_adjusted, dataUtil.convert_to_annotation_rects_label(ag_adjusted[i], labels_result[i]), (0, 255, 255))
        image_adjusted.resize(output_image_size, Image.ANTIALIAS).save('test_images/{}_labels_adjusted.jpg'.format(i))        

        dataUtil.draw_bounding_boxes(image_adjusted, dataUtil.convert_to_annotation_rects_output(ag_adjusted[i], probabilities_predicted[i]), (0, 0, 255))
        image_adjusted.resize(output_image_size, Image.ANTIALIAS).save('test_images/{}_estimates_adjusted.jpg'.format(i))


        test_paths.append('test_images/{}_estimates.jpg'.format(i))

    # Saving detections for evaluation purposes
    evaluation.prepare_detections(probabilities_predicted, anchor_grid, test_paths, batch_size)