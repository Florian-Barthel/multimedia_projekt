from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import config
import dataUtil
import graph
import evaluation
import fileUtil
import eval_script.eval_detections_own as validation
import mobilenet
import draw
import dataSet_new

fileUtil.run_tensorboard()

anchor_grid_tensor = tf.constant(config.anchor_grid, dtype=tf.int32)

train_dataset = dataSet_new.create(config.train_dataset, config.batch_size)

handle = tf.placeholder(tf.string, shape=[], name='handle')
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

train_iterator = train_dataset.make_one_shot_iterator()

next_element = iterator.get_next()

with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())

    images_tensor, gt_tensor, labels_tensor = next_element

    features = mobilenet.mobile_net_v2()(images_tensor, training=False)

    probabilities = graph.probabilities_output(features)
    probabilities_loss, num_labels, num_weights, num_predicted = graph.probabilities_loss(probabilities, labels_tensor)

    adjustments = graph.adjustments_output(features)
    adjustments_loss = graph.adjustments_loss(adjustments, gt_tensor, labels_tensor, anchor_grid_tensor)
    anchor_grid_adjusted = dataUtil.calculate_adjusted_anchor_grid(anchor_grid_tensor, adjustments)

    if config.use_bounding_box_regression:
        total_loss = probabilities_loss * adjustments_loss
    else:
        total_loss = probabilities_loss
    probabilities_loss = tf.identity(probabilities_loss, name='probabilities_loss')
    adjustments_loss = tf.identity(adjustments_loss, name='adjustments_loss')
    total_loss = tf.identity(total_loss, name='total_loss')

    def optimize(target_loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        objective = optimizer.minimize(loss=target_loss, global_step=config.global_step, name='optimize')
        return objective
        

    optimize = optimize(total_loss)

    # Tensorboard config
    tf.summary.scalar('1_losses/total', total_loss)
    tf.summary.scalar('1_losses/probabilities', probabilities_loss)
    tf.summary.scalar('1_losses/adjustments', adjustments_loss)
    # py_mAP = 0
    mAPs_tensor = tf.placeholder(shape=[2], dtype=tf.float32, name='mAPs_tensor')
    tf.summary.scalar('2_mAP/probabilities', mAPs_tensor[0])
    tf.summary.scalar('2_mAP/total', mAPs_tensor[1])

    tf.summary.scalar('3_debug/Number of positives', num_predicted)

    merged_summary = tf.summary.merge_all(name='merged_summary')
    log_writer = tf.summary.FileWriter(config.logs_directory, sess.graph, flush_secs=1)
    saver = tf.train.Saver()

    graph_vars = tf.compat.v1.global_variables()
    progress_bar_init = tqdm(graph_vars)
    progress_bar_init.set_description('INIT  | ')
    vars_to_init = []
    for var in progress_bar_init:
        try:
            sess.run(var)
        except:
            vars_to_init.append(var)

    sess.run(tf.compat.v1.variables_initializer(vars_to_init))

    validation_data = dataUtil.get_validation_data(100, config.anchor_grid)


    mAPs = [0.0, 0.0]
    progress_bar_train = tqdm(range(config.iterations))
    for i in progress_bar_train:
        progress_bar_train.set_description('TRAIN | ')
        loss, labels, weights, predicted, _, summary = sess.run(
            [[total_loss, probabilities_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize,
             merged_summary],
            feed_dict={handle: train_handle, mAPs_tensor: mAPs})  # ,
        # mAP: py_mAP})
        if i % 10 == 0:
            log_writer.add_summary(summary, i)

        '''
        Run validation every 500 iterations
        '''

        if (i + 1) % config.validation_interval == 0:
            fileUtil.update_directories(i)

            detection_path_normal = config.iteration_directory + 'detection_normal.txt'
            detection_path_bbr = config.iteration_directory + 'detection_bbr.txt'
            print('\nvalidation...')
            for j in range(len(validation_data)):
                (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[j]
                probabilities_output, anchor_grid_bbr_output = sess.run([probabilities, anchor_grid_adjusted],
                                                                        feed_dict={images_tensor: test_images,
                                                                                   labels_tensor: test_labels,
                                                                                   mAPs_tensor: mAPs})
                evaluation.prepare_detections(probabilities_output, config.anchor_grid, test_paths, detection_path_normal)

                if config.use_bounding_box_regression:
                    evaluation.prepare_detections(probabilities_output, anchor_grid_bbr_output, test_paths,
                                                  detection_path_bbr)
            py_mAP_normal = validation.run(
                detection_file=detection_path_normal,
                result_file=config.iteration_directory + 'normal',
                dataset_dir=config.validation_directory) * 100
            print('mAP static anchor grid:', py_mAP_normal)

            mAPs[0] = py_mAP_normal

            if config.use_bounding_box_regression:
                py_mAP_bbr = validation.run(
                    detection_file=detection_path_bbr,
                    result_file=config.iteration_directory + 'bbr',
                    dataset_dir=config.validation_directory) * 100
                print('mAP bounding box regression:', py_mAP_bbr)

            mAPs[1] = py_mAP_bbr
            fileUtil.save_model(saver, sess)

            # str_summary = sess.run(merged_summary, feed_dict={mAP: str(py_mAP)})
            # log_writer.add_summary(str_summary, i)

            draw.draw_images(num_images=5,
                             images=test_images,
                             output=probabilities_output,
                             labels=test_labels,
                             gts=gt_annotation_rects,
                             adjusted_anchor_grid=anchor_grid_bbr_output,
                             original_anchor_grid=config.anchor_grid,
                             path=config.iteration_directory)