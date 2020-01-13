import os
import numpy as np
from PIL import Image
import config
import dataUtil
from datetime import datetime
from tensorboard import program
import tensorflow as tf
from tqdm import tqdm

image_path = 'test_images'
max_drawn_images = 10
fg_threshold = 0.7

# TensorBoard logs saved in ./logs/dd-MM-yyyy_HH-mm-ss
current_time = datetime.now()
logs_directory = './logs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S')
detection_directory = 'eval_script/detections/' + current_time.strftime('%d-%m-%Y_%H-%M-%S') + '/'
validation_directory = 'dataset_mmp'
model_directory = './models/'

if not os.path.exists(detection_directory):
    os.makedirs(detection_directory)

def save_model(saver, sess):
    saver.save(sess, model_directory + current_time.strftime('%d-%m-%Y_%H-%M-%S') + '/model')

def load_model(model_path, sess):
    saver = tf.train.import_meta_graph(model_path + "model.meta")
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    return saver

def draw_images(test_images, test_labels, anchor_grid, ag_adjusted_output, probabilities_output):
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
    num_view_images = 30
    progress_bar_save = tqdm(range(num_view_images))
    progress_bar_save.set_description('SAVE  | ')
    for i in progress_bar_save:
        image = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        image.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_gts.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_label(anchor_grid, test_labels[i]),
                                     (0, 255, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_labels.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image,
                                     dataUtil.convert_to_annotation_rects_output(anchor_grid, probabilities_output[i]),
                                     (0, 0, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save('test_images/{}_estimates.jpg'.format(i))

        image_adjusted = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')

        dataUtil.draw_bounding_boxes(image_adjusted,
                                     dataUtil.convert_to_annotation_rects_label(ag_adjusted_output[i], test_labels[i]),
                                     (0, 255, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(
            'test_images/{}_labels_adjusted.jpg'.format(i))

        dataUtil.draw_bounding_boxes(image_adjusted, dataUtil.convert_to_annotation_rects_output(ag_adjusted_output[i],
                                                                                                 probabilities_output[
                                                                                                     i]), (0, 0, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(
            'test_images/{}_estimates_adjusted.jpg'.format(i))


def run_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir=logs'])
    url = tb.launch()
