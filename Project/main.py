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

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

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


    overlap_labels_tensor = dataUtil.calculate_overlap_boxes_tensor(gt_labels_tensor, anchor_grid, iou)

    calculate_output, calculate_offsets = graph.output(images=images_tensor,
                                    num_scales=len(scales),
                                    num_aspect_ratios=len(aspect_ratios),
                                    f_rows=f_map_rows,
                                    f_cols=f_map_cols)


    calculate_loss, num_labels, num_random, num_weights, num_predicted = graph.loss(input_tensor=calculate_output,
                                                                                    labels=overlap_labels_tensor,
                                                                                    negative_percentage=negative_percentage)



    calculate_offsets_loss = graph.offsets_loss(calculate_offsets, gt_labels_tensor, tf.constant(anchor_grid, dtype=tf.int32))
    
    def optimize(my_loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        objective = optimizer.minimize(loss=my_loss)
        return objective


    optimize = optimize(calculate_loss)

    tf.summary.scalar('loss', calculate_loss)
    tf.summary.scalar('num_predicted', num_predicted)
    merged_summary = tf.summary.merge_all()

    graph_vars = tf.global_variables()
    for var in tqdm(graph_vars):
        try:
            sess.run(var)
        except:
            print('found uninitialized variable {}'.format(var.name))
            sess.run(tf.initialize_variables([var]))

    # TensorBoard graph summary
    log_writer = tf.summary.FileWriter(logs_directory, sess.graph, flush_secs=5)
    progress_bar = tqdm(range(iterations))
    for i in progress_bar:

        loss, labels, random, weights, predicted, _, summary, _ = sess.run([calculate_loss, num_labels, num_random, num_weights, num_predicted, optimize, merged_summary, calculate_offsets_loss], feed_dict={handle: train_handle})

        description = ' loss:' + str(np.around(loss, decimals=5)) + ' num_labels: ' + str(
            labels) + ' num_random: ' + str(random) + ' num_weights: ' + str(weights) + ' num_predicted: ' + str(
            predicted)
        progress_bar.set_description(description, refresh=True)
        # TensorBoard scalar summary
        log_writer.add_summary(summary, i)



    gt_images_tensor = tf.image.draw_bounding_boxes(images_tensor, gt_labels_tensor)
    images_result, labels_result, output_result = sess.run([gt_images_tensor, overlap_labels_tensor, calculate_output], feed_dict={handle: test_handle})

    test_paths = []
    for i in range(np.shape(images_result)[0]):
        image = Image.fromarray(((images_result[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        image.resize((720, 720), Image.ANTIALIAS).save('test_images/{}_gts.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image=image,
                             annotation_rects=dataUtil.convert_to_annotation_rects_label(anchor_grid, labels_result[i]),
                             color=(0, 255, 255))

        image.resize((720, 720), Image.ANTIALIAS).save('test_images/{}_labels.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image=image,
                                     annotation_rects=dataUtil.convert_to_annotation_rects_output(anchor_grid, output_result[i]),
                                     color=(0, 0, 255))
        
        image.resize((720, 720), Image.ANTIALIAS).save('test_images/{}_estimates.jpg'.format(i))
        test_paths.append('test_images/{}_estimates.jpg'.format(i))

    # Saving detections for evaluation purposes
    evaluation.prepare_detections(output_result, anchor_grid, test_paths, batch_size)
