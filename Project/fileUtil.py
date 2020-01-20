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

def run_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir=logs'])
    url = tb.launch()

def update_directories(iteration):
    if not os.path.exists(config.run_directory):
        os.makedirs(config.run_directory)
    config.iteration_directory = config.run_directory + str(iteration) + '/'
    config.model_directory = config.iteration_directory + 'model/'
    if not os.path.exists(config.iteration_directory):
        os.makedirs(config.iteration_directory)
    if not os.path.exists(config.model_directory):
        os.makedirs(config.model_directory)

def save_model(saver, sess):
    print('Saving model to: ' + config.model_directory)
    saver.save(sess, config.model_directory + 'model')

def load_model(model_path, sess):
    print('Loading model from: ' + model_path)
    saver = tf.train.import_meta_graph(model_path + "model.meta")
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    return saver

def get_latest_model_path():
    runs = os.listdir(config.runs)
    if not runs:
        print("No runs found at " + config.runs)
        exit(-1)
    run_dates = []
    for r in runs:
        run_dates.append(datetime.strptime(r, '%d-%m-%Y_%H-%M-%S'))
    run_dates.sort(reverse=True)
    latest_run_path = config.runs + datetime.strftime(run_dates[0], '%d-%m-%Y_%H-%M-%S') + '/'
    models = os.listdir(latest_run_path)
    if not models:
        print("No models found at " + latest_run_path)
        exit(-1)
    latest_model_path = latest_run_path + max(models) + '/model/'
    return latest_model_path

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
