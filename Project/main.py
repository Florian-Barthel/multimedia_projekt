from tqdm import tqdm
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import dataSet
import config
import dataUtil
import graph
import anchorgrid
import evaluation
import fileUtil
import eval_script.eval_detections_own as validation
import mobilenet

fileUtil.run_tensorboard()

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
    log_writer = tf.summary.FileWriter(fileUtil.logs_directory, sess.graph, flush_secs=1)

    saver = tf.train.Saver()

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
        if (i + 1) % 501 == 0:
            detection_file = str(i) + '_detection.txt'
            print('validation...')
            for j in range(len(validation_data)):
                (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[j]
                probabilities_output, ag_adjusted_output = sess.run([probabilities, ag_adjusted_tensor], feed_dict={images_tensor: test_images})
                evaluation.prepare_detections(probabilities_output, ag_adjusted_output, test_paths, fileUtil.detection_directory + detection_file)
            mAP = validation.run(fileUtil.detection_directory + detection_file, fileUtil.detection_directory + str(i),
                                 fileUtil.validation_directory) * 100
            print('mAP: ' + str(mAP))
            print('Saving model...')
            fileUtil.save_model(saver, sess)
        log_writer.add_summary(summary, i)

    progress_bar_validate = tqdm(range(len(validation_data)))
    progress_bar_validate.set_description('VAL   | ')
    for i in progress_bar_validate:
        (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[i]
        probabilities_output, ag_adjusted_output = sess.run([probabilities, ag_adjusted_tensor], feed_dict={images_tensor: test_images})
        evaluation.prepare_detections(probabilities_output, ag_adjusted_output, test_paths, fileUtil.detection_directory + str(config.iterations) + '_detection.txt')

    fileUtil.draw_images(test_images, test_labels, anchor_grid, ag_adjusted_output, probabilities_output)

