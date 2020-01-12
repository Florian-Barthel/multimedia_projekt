from tqdm import tqdm
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import dataSet
import config
import dataUtil
from datetime import datetime
import graph
import anchorgrid
import evaluation
import eval_script.eval_detections_own as validation
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

# TensorBoard logs saved in ./logs/dd-MM-yyyy_HH-mm-ss
current_time = datetime.now()
logs_directory = './logs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S')
detection_directory = 'eval_script/detections/' + current_time.strftime('%d-%m-%Y_%H-%M-%S') + '/'
validation_directory = 'dataset_mmp'

if not os.path.exists(detection_directory):
    os.makedirs(detection_directory)

anchor_grid = anchorgrid.anchor_grid(f_map_rows=config.f_map_rows,
                                     f_map_cols=config.f_map_cols,
                                     scale_factor=config.scale_factor,
                                     scales=config.scales,
                                     aspect_ratios=config.aspect_ratios)

train_dataset = dataSet.create("./dataset_2_crowd_min ", anchor_grid).batch(
    config.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())

    images_tensor, labels_tensor = next_element

    # use only raw images!
    # shape (batch_size, 320, 320, 3)
    no_gts_images_tensor = images_tensor[:, 0]

    calculate_output = graph.output(input_tensor=no_gts_images_tensor)

    calculate_loss, num_labels, num_weights, num_predicted = graph.loss(input_tensor=calculate_output,
                                                                        labels_tensor=labels_tensor)


    def optimize(my_loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
        objective = optimizer.minimize(loss=my_loss)
        return objective


    optimize = optimize(calculate_loss)

    '''
    Tensorboard config
    '''
    tf.summary.scalar('loss', calculate_loss)
    tf.summary.scalar('num_predicted', num_predicted)
    mAP = 0
    tf.summary.scalar('mAP', mAP)
    merged_summary = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(logs_directory, sess.graph)

    graph_vars = tf.compat.v1.global_variables()
    progress_bar_init = tqdm(graph_vars)
    progress_bar_init.set_description('INIT  | ')
    for var in progress_bar_init:
        try:
            sess.run(var)
        except:
            sess.run(tf.compat.v1.variables_initializer([var]))

    validation_data = dataUtil.get_validation_data(200, anchor_grid)

    progress_bar_train = tqdm(range(config.iterations))
    for i in progress_bar_train:
        progress_bar_train.set_description('TRAIN | ')
        loss, labels, weights, predicted, _, summary = sess.run(
            [calculate_loss, num_labels, num_weights, num_predicted, optimize, merged_summary],
            feed_dict={handle: train_handle})

        '''
        Run validation every 500 iterations
        '''
        # TODO: Save model
        if (i + 1) % 501 == 0:
            detection_file = str(i) + '_detection.txt'
            print('validation...')
            for j in range(len(validation_data)):
                (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[j]
                output = sess.run(calculate_output, feed_dict={no_gts_images_tensor: test_images})
                evaluation.prepare_detections(output, anchor_grid, test_paths, detection_directory + detection_file)
            mAP = validation.run(detection_directory + detection_file, detection_directory + str(i), validation_directory) * 100
            print('mAP: ' + str(mAP))
        log_writer.add_summary(summary, i)

    progress_bar_validate = tqdm(range(len(validation_data)))
    progress_bar_validate.set_description('VAL   | ')
    for i in progress_bar_validate:
        (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[i]
        output = sess.run(calculate_output, feed_dict={no_gts_images_tensor: test_images})
        evaluation.prepare_detections(output, anchor_grid, test_paths, detection_directory + str(config.iterations) + '_detection.txt')

    num_view_images = 30
    progress_bar_save = tqdm(range(num_view_images))
    progress_bar_save.set_description('SAVE  | ')
    for i in progress_bar_save:
        img = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        dataUtil.draw_bounding_boxes(image=img,
                                     annotation_rects=dataUtil.convert_to_annotation_rects_label(anchor_grid,
                                                                                                 test_labels[i]),
                                     color=(255, 100, 100))
        dataUtil.draw_bounding_boxes(image=img,
                                     annotation_rects=dataUtil.convert_to_annotation_rects_output(anchor_grid,
                                                                                                  output[i]),
                                     color=(100, 255, 100))
        dataUtil.draw_bounding_boxes(image=img,
                                     annotation_rects=gt_annotation_rects[i],
                                     color=(100, 100, 255))
        if not os.path.exists('test_images'):
            os.makedirs('test_images')
        img.save('test_images/max_overlap_boxes_{}.jpg'.format(i))
