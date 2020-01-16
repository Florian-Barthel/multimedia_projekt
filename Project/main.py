from tqdm import tqdm
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
import eval_script.eval_detections_own as validation
from tensorboard import program
import mobilenet
import draw

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

if not os.path.exists(config.detection_directory):
    os.makedirs(config.detection_directory)

anchor_grid = anchorgrid.anchor_grid(f_map_rows=config.f_map_rows,
                                     f_map_cols=config.f_map_cols,
                                     scale_factor=config.scale_factor,
                                     scales=config.scales,
                                     aspect_ratios=config.aspect_ratios)

anchor_grid_tensor = tf.constant(anchor_grid, dtype=tf.int32)

train_dataset = dataSet.create(config.train_dataset, config.batch_size)

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
    anchor_grid_adjusted = dataUtil.calculate_adjusted_anchor_grid(anchor_grid_tensor, adjustments)

    if config.use_bounding_box_regression:
        total_loss = probabilities_loss + adjustments_loss
    else:
        total_loss = probabilities_loss


    def optimize(target_loss):
        if config.use_adam_optimizer:
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            objective = optimizer.minimize(loss=target_loss, global_step=config.global_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
            objective = optimizer.minimize(target_loss)
        return objective


    optimize = optimize(total_loss)

    # Tensorboard config
    tf.summary.scalar('losses/total', total_loss)
    tf.summary.scalar('losses/probabilities', probabilities_loss)
    tf.summary.scalar('losses/adjustments', adjustments_loss)
    # py_mAP = 0
    # mAP = tf.placeholder(shape=(), dtype=tf.float32)
    # tf.summary.scalar('score/mAP', mAP)

    tf.summary.scalar('debug/Number of positivies', num_predicted)

    merged_summary = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(config.logs_directory, sess.graph, flush_secs=1)

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

    validation_data = dataUtil.get_validation_data(100, anchor_grid)

    progress_bar_train = tqdm(range(config.iterations))
    for i in progress_bar_train:
        progress_bar_train.set_description('TRAIN | ')
        loss, labels, weights, predicted, _, summary = sess.run(
            [[total_loss, probabilities_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize,
             merged_summary],
            feed_dict={handle: train_handle})  # ,
        # mAP: py_mAP})
        if i % 10 == 0:
            log_writer.add_summary(summary, i)

        '''
        Run validation every 500 iterations
        '''
        if (i + 1) % config.validation_interval == 0:
            iteration_directory = config.detection_directory + '/' + str(i) + '/'
            if not os.path.exists(iteration_directory):
                os.makedirs(iteration_directory)

            detection_path_normal = iteration_directory + 'detection_bbr.txt'
            detection_path_bbr = iteration_directory + 'detection_normal.txt'
            print('\nvalidation...')
            for j in range(len(validation_data)):
                (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[j]
                probabilities_output, anchor_grid_bbr_output = sess.run([probabilities, anchor_grid_adjusted],
                                                                        feed_dict={images_tensor: test_images})

                evaluation.prepare_detections(probabilities_output, anchor_grid, test_paths, detection_path_normal)

                if config.use_bounding_box_regression:
                    evaluation.prepare_detections(probabilities_output, anchor_grid_bbr_output, test_paths,
                                                  detection_path_bbr)

            py_mAP_normal = validation.run(
                detection_file=detection_path_normal,
                result_file=iteration_directory + 'normal',
                dataset_dir=config.validation_directory) * 100
            print('mAP static anchor grid:', py_mAP_normal)

            if config.use_bounding_box_regression:
                py_mAP_bbr = validation.run(
                    detection_file=detection_path_bbr,
                    result_file=iteration_directory + 'bbr',
                    dataset_dir=config.validation_directory) * 100
                print('mAP bounding box regression:', py_mAP_bbr)

            # str_summary = sess.run(merged_summary, feed_dict={mAP: str(py_mAP)})
            # log_writer.add_summary(str_summary, i)

            draw.draw_images(num_images=5,
                             images=test_images,
                             output=probabilities_output,
                             labels=test_labels,
                             gts=gt_annotation_rects,
                             adjusted_anchor_grid=anchor_grid_bbr_output,
                             original_anchor_grid=anchor_grid,
                             path=iteration_directory)
            # TODO: Save model
