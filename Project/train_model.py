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
import argparse as ap

# Trains the specified model
# if no arguments are given: trains the latest model
if __name__ == '__main__':
    cparse = ap.ArgumentParser(prog='train_model', description='Trains an existing model')
    cparse.add_argument('--model', help='Path to the model', default='')
    args = cparse.parse_args()
    if args.model == '':
        model_path = fileUtil.get_latest_model_path()
    else:
        model_path = args.model

    if not model_path.endswith('/'):
        model_path += '/'
    config.model_directory = model_path

    fileUtil.run_tensorboard()

    with tf.Session() as sess:
        saver = fileUtil.load_model(model_path=model_path, sess=sess)
        graph = tf.get_default_graph()
        for tensor in graph.get_operations():
            print(tensor.name)

        total_loss = graph.get_tensor_by_name('total_loss:0')
        probabilities_loss = graph.get_tensor_by_name('probabilities_loss:0')
        adjustments_loss = graph.get_tensor_by_name('adjustments_loss:0')
        num_labels = graph.get_tensor_by_name('num_labels:0')
        num_weights = graph.get_tensor_by_name('num_weights:0')
        num_predicted = graph.get_tensor_by_name('num_predicted:0')
        optimize = graph.get_tensor_by_name('optimize:0')
        merged_summary = graph.get_tensor_by_name('merged_summary/merged_summary:0')
        handle = graph.get_tensor_by_name('handle:0')
        mAPs_tensor = graph.get_tensor_by_name('mAPs_tensor:0')
        probabilities = graph.get_tensor_by_name('probabilities/probabilities:0')
        anchor_grid_adjusted = graph.get_tensor_by_name('anchor_grid_adjusted:0')

        train_dataset = dataSet_new.create(config.train_dataset, config.batch_size)
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        train_handle = sess.run(train_iterator.string_handle())
        images_tensor, gt_tensor, labels_tensor = next_element

        log_writer = tf.summary.FileWriter(config.logs_directory, sess.graph, flush_secs=1)

        validation_data = dataUtil.get_validation_data(config.batch_size, config.anchor_grid)
        mAPs = [0.0, 0.0]
        progress_bar_train = tqdm(range(config.train_iterations))
        for i in progress_bar_train:
            progress_bar_train.set_description('TRAIN | ')
            loss, labels, weights, predicted, _, summary = sess.run(
                [[total_loss, probabilities_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize,
                 merged_summary],
                feed_dict={handle: train_handle, mAPs_tensor: mAPs})
            if i % 10 == 0:
                log_writer.add_summary(summary, i)

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
                                                                                       mAPs_tensor: mAPs,
                                                                                       handle: train_handle})
                    evaluation.prepare_detections(probabilities_output, config.anchor_grid, test_paths,
                                                  detection_path_normal)

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

                draw.draw_images(num_images=5,
                                 images=test_images,
                                 output=probabilities_output,
                                 labels=test_labels,
                                 gts=gt_annotation_rects,
                                 adjusted_anchor_grid=anchor_grid_bbr_output,
                                 original_anchor_grid=config.anchor_grid,
                                 path=config.iteration_directory)

        fileUtil.save_model(saver=saver, sess=sess)
