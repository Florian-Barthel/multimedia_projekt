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
        total_loss = graph.get_tensor_by_name('total_loss:0')
        #probabilities_loss = graph.get_tensor_by_name('probabilities_loss:0')
        adjustments_loss = graph.get_tensor_by_name('adjustments_loss:0')
        num_labels = graph.get_tensor_by_name('num_labels:0')
        num_weights = graph.get_tensor_by_name('num_weights:0')
        num_predicted = graph.get_tensor_by_name('num_predicted:0')
        optimize = graph.get_tensor_by_name('optimize:0')
        merged_summary = graph.get_tensor_by_name('merged_summary/merged_summary:0')
        handle = graph.get_tensor_by_name('handle:0')
        mAPs_tensor = graph.get_tensor_by_name('mAPs_tensor:0')

        mAPs = [0.0, 0.0]
        train_dataset = dataSet_new.create(config.train_dataset, config.batch_size)
        train_iterator = train_dataset.make_one_shot_iterator()
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        next_element = iterator.get_next()
        train_handle = sess.run(train_iterator.string_handle())
        progress_bar_train = tqdm(range(config.iterations))
        for i in progress_bar_train:
            progress_bar_train.set_description('TRAIN | ')
            loss, labels, weights, predicted, _, summary = sess.run(
                [[total_loss, total_loss, adjustments_loss], num_labels, num_weights, num_predicted, optimize,
                 merged_summary],
                feed_dict={handle: train_handle, mAPs_tensor: mAPs})  # ,
            # mAP: py_mAP})
        print(loss)

        fileUtil.save_model(saver=saver, sess=sess)
