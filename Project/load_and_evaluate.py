from tqdm import tqdm
import os
import argparse as ap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import config
import dataUtil
import graph
import evaluation
import eval_script.eval_detections_own as validation
import mobilenet
import draw
from datetime import datetime

cparse = ap.ArgumentParser(
        prog='Prepare evaluation',
        description='Prepares evaluation txt file and runs evaluation.')
cparse.add_argument('--val_data', help='Path to validation data. Default: dataset_mmp/test', default='dataset_mmp/test')

cmdargs = cparse.parse_args()

# validation_directory = 'asdf'
# validation_directory = 'dataset_mmp/test'
validation_directory = str(cmdargs.val_data)
model_directory = 'runs/28-01-2020_00-44-35_best/89999'

current_time = datetime.now()
result_directory = 'results/' + current_time.strftime('%d-%m-%Y_%H-%M-%S') + '/'
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

anchor_grid_tensor = tf.constant(config.anchor_grid, dtype=tf.int32)

images_tensor = tf.placeholder(tf.float32, [None, 320, 320, 3])
'''
BUILD GRAPH
'''

with tf.Session() as sess:
    features = mobilenet.mobile_net_v2()(images_tensor, training=False)
    probabilities = graph.probabilities_output(features)
    adjustments = graph.adjustments_output(features)
    anchor_grid_adjusted = dataUtil.calculate_adjusted_anchor_grid(anchor_grid_tensor, adjustments)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(model_directory))

    validation_data = dataUtil.get_validation_data(100, config.anchor_grid, validation_directory)

    '''
    VALIDATION
    '''
    detection_path_bbr = result_directory + 'detection_bbr.txt'
    print('\nvalidation...')
    for j in tqdm(range(len(validation_data))):
        (test_images, test_labels, gt_annotation_rects, test_paths) = validation_data[j]
        probabilities_output, anchor_grid_bbr_output = sess.run([probabilities, anchor_grid_adjusted],
                                                                feed_dict={images_tensor: test_images})

        evaluation.prepare_detections(probabilities_output, anchor_grid_bbr_output, test_paths,
                                      detection_path_bbr, fg_threshold=0)

    py_mAP_bbr = validation.run(
        detection_file=detection_path_bbr,
        result_file=result_directory + 'bbr',
        dataset_dir=config.validation_directory) * 100
    print('mAP bounding box regression:', py_mAP_bbr)

    draw.draw_images(num_images=5,
                     images=test_images,
                     output=probabilities_output,
                     labels=test_labels,
                     gts=gt_annotation_rects,
                     adjusted_anchor_grid=anchor_grid_bbr_output,
                     original_anchor_grid=config.anchor_grid,
                     path=result_directory)

