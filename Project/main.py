from tqdm import tqdm
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import config
import dataUtil
import graph
import evaluation
import eval_script.eval_detections_own as validation
from tensorboard import program
import mobilenet
import draw
import dataSet_new

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=logs'])
url = tb.launch()

if not os.path.exists(config.detection_directory):
    os.makedirs(config.detection_directory)

anchor_grid_tensor = tf.constant(config.anchor_grid, dtype=tf.int32)

train_dataset = dataSet_new.create(config.train_dataset, config.batch_size)
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
train_iterator = train_dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
    train_handle = sess.run(train_iterator.string_handle())
    images_tensor, gt_tensor, labels_tensor = next_element

    '''
    BUILD GRAPH
    '''
    features = mobilenet.mobile_net_v2()(images_tensor, training=False)
    probabilities = graph.probabilities_output(features)
    adjustments = graph.adjustments_output(features)

    probabilities_loss, num_labels, num_weights, num_predicted = graph.probabilities_loss(probabilities, labels_tensor)
    adjustments_loss = graph.adjustments_loss(adjustments, gt_tensor, labels_tensor, anchor_grid_tensor)

    anchor_grid_adjusted = dataUtil.calculate_adjusted_anchor_grid(anchor_grid_tensor, adjustments)

    if config.use_bounding_box_regression:
        # TODO: Change back
        total_loss = probabilities_loss * adjustments_loss
    else:
        total_loss = probabilities_loss


    def optimize(target_loss):
        if config.use_adam_optimizer:
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
            objective = optimizer.minimize(loss=target_loss, global_step=config.global_step)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
            objective = optimizer.minimize(loss=target_loss)
        return objective


    saver = tf.compat.v1.train.Saver(max_to_keep=None)
    optimize = optimize(total_loss)

    '''
    TENSORBOARD CONFIGURATION
    '''
    tf.summary.scalar('1_losses/total', total_loss)
    tf.summary.scalar('1_losses/probabilities', probabilities_loss)
    tf.summary.scalar('1_losses/adjustments', adjustments_loss)

    mAPs_tensor = tf.placeholder(shape=[2], dtype=tf.float32)
    tf.summary.scalar('2_mAP/probabilities', mAPs_tensor[0])
    tf.summary.scalar('2_mAP/total', mAPs_tensor[1])

    tf.summary.scalar('3_debug/Number of positives', num_predicted)

    merged_summary = tf.summary.merge_all()
    log_writer = tf.summary.FileWriter(config.logs_directory, sess.graph, flush_secs=1)

    '''
    INITIALIZE VARS
    '''
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
        '''
        TRAINING
        '''
        progress_bar_train.set_description('TRAIN | ')
        loss, labels, weights, predicted, _, summary = sess.run(
            [[total_loss, probabilities_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize,
             merged_summary],
            feed_dict={handle: train_handle, mAPs_tensor: mAPs})
        if i % 20 == 0:
            log_writer.add_summary(summary, i)

        '''
        VALIDATION
        '''
        if (i + 1) % config.validation_interval == 0:
            iteration_directory = config.detection_directory + '/' + str(i) + '/'
            if not os.path.exists(iteration_directory):
                os.makedirs(iteration_directory)

            detection_path_normal = iteration_directory + 'detection_normal.txt'
            detection_path_bbr = iteration_directory + 'detection_bbr.txt'
            print('\nvalidation...')
            for j in range(len(validation_data)):
                (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[j]
                probabilities_output, anchor_grid_bbr_output = sess.run([probabilities, anchor_grid_adjusted],
                                                                        feed_dict={images_tensor: test_images,
                                                                                   labels_tensor: test_labels,
                                                                                   mAPs_tensor: mAPs})

                evaluation.prepare_detections(probabilities_output, config.anchor_grid, test_paths,
                                              detection_path_normal)

                if config.use_bounding_box_regression:
                    evaluation.prepare_detections(probabilities_output, anchor_grid_bbr_output, test_paths,
                                                  detection_path_bbr)

            py_mAP_normal = validation.run(
                detection_file=detection_path_normal,
                result_file=iteration_directory + 'normal',
                dataset_dir=config.validation_directory) * 100
            print('mAP static anchor grid:', py_mAP_normal)
            mAPs[0] = py_mAP_normal

            if config.use_bounding_box_regression:
                py_mAP_bbr = validation.run(
                    detection_file=detection_path_bbr,
                    result_file=iteration_directory + 'bbr',
                    dataset_dir=config.validation_directory) * 100
                print('mAP bounding box regression:', py_mAP_bbr)
                mAPs[1] = py_mAP_bbr

            draw.draw_images(num_images=5,
                             images=test_images,
                             output=probabilities_output,
                             labels=test_labels,
                             gts=gt_annotation_rects,
                             adjusted_anchor_grid=anchor_grid_bbr_output,
                             original_anchor_grid=config.anchor_grid,
                             path=iteration_directory)
            '''
            Save Model
            '''

            saver.save(sess, iteration_directory + 'model', global_step=i)
