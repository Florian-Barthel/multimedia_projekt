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
import mobilenet

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

anchor_grid_tensor = tf.constant(anchor_grid, dtype=tf.int32)

train_dataset = dataSet.create("./dataset_mmp/train", config.batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())

    images_tensor, gt_labels_tensor = next_element

    labels_tensor = dataUtil.calculate_overlap_boxes_tensor(gt_labels_tensor, anchor_grid)

    features = mobilenet.mobile_net_v2()(images_tensor)

    probabilities = graph.probabilities_output(features)
    probabilities_loss, num_labels, num_weights, num_predicted = graph.probabilities_loss(probabilities, labels_tensor)
    
    adjustments = graph.adjustments_output(features)
    adjustments_loss = graph.adjustments_loss(adjustments, gt_labels_tensor, labels_tensor, anchor_grid_tensor)
    ag_adjusted_tensor = dataUtil.calculate_adjusted_anchor_grid(anchor_grid_tensor, adjustments)

    total_loss = probabilities_loss * adjustments_loss
    

    def optimize(target_loss):
        optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        objective = optimizer.minimize(target_loss)
        return objective

    optimize = optimize(total_loss)
    

    #Tensorboard config
    tf.summary.scalar('losses/total', total_loss)
    tf.summary.scalar('losses/probabilities', probabilities_loss)
    tf.summary.scalar('losses/adjustments', adjustments_loss)

    mAP = 0
    tf.summary.scalar('score/mAP', mAP)

    tf.summary.scalar('debug/Number of positivies', num_predicted)


    merged_summary = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(logs_directory, sess.graph, flush_secs=1)

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
            [[total_loss, probabilities_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize, merged_summary],
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
                probabilites_output, ag_adjusted_output = sess.run([probabilities, ag_adjusted_tensor], feed_dict={images_tensor: test_images})
                evaluation.prepare_detections(probabilites_output, ag_adjusted_output, test_paths, detection_directory + detection_file)
            mAP = validation.run(detection_directory + detection_file, detection_directory + str(i), validation_directory) * 100
            print('mAP: ' + str(mAP))
        log_writer.add_summary(summary, i)

    progress_bar_validate = tqdm(range(len(validation_data)))
    progress_bar_validate.set_description('VAL   | ')
    for i in progress_bar_validate:
        (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[i]
        probabilites_output, ag_adjusted_output = sess.run([probabilities, ag_adjusted_tensor], feed_dict={images_tensor: test_images})
        evaluation.prepare_detections(probabilites_output, ag_adjusted_output, test_paths, detection_directory + str(config.iterations) + '_detection.txt')



    if not os.path.exists('test_images'):
        os.makedirs('test_images')
    num_view_images = 30
    progress_bar_save = tqdm(range(num_view_images))
    progress_bar_save.set_description('SAVE  | ')
    for i in progress_bar_save:
        image = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        image.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_gts.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_label(anchor_grid, test_labels[i]), (0, 255, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_labels.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_output(anchor_grid, probabilites_output[i]), (0, 0, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_estimates.jpg'.format(i))


        image_adjusted = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        
        dataUtil.draw_bounding_boxes(image_adjusted, dataUtil.convert_to_annotation_rects_label(ag_adjusted_output[i], test_labels[i]), (0, 255, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_labels_adjusted.jpg'.format(i))        

        dataUtil.draw_bounding_boxes(image_adjusted, dataUtil.convert_to_annotation_rects_output(ag_adjusted_output[i], probabilites_output[i]), (0, 0, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_estimates_adjusted.jpg'.format(i))
