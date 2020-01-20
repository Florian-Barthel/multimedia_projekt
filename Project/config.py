import tensorflow as tf
from datetime import datetime
import anchorgrid

use_decaying_learning_rate = True
use_augmentation = True
use_hard_negative_mining = False
use_bounding_box_regression = True
use_different_dataset = True

image_width = 320
image_height = 320
output_image_size = (720, 720)

crop_factor = 0.1
augmentation_probability = 0.15

f_map_rows = 10
f_map_cols = 10
scales = [80, 100, 150]
aspect_ratios = [0.5, 1.0, 2.0]
scale_factor = 32.0

anchor_grid = anchorgrid.anchor_grid(f_map_rows=f_map_rows,
                                     f_map_cols=f_map_cols,
                                     scale_factor=scale_factor,
                                     scales=scales,
                                     aspect_ratios=aspect_ratios)

batch_size = 16
iterations = 100000
validation_interval = 500

iou = 0.5
nms_threshold = 0.3

negative_example_factor = 10

detection_foreground_threshold = 0.05

# TensorBoard logs saved in ./logs/dd-MM-yyyy_HH-mm-ss
current_time = datetime.now()
logs_directory = './logs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S')
detection_directory = 'runs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S') + '/'
validation_directory = 'dataset_mmp'

learning_rate = 0.0001

global_step = tf.Variable(0, trainable=False)
if use_decaying_learning_rate:
    starter_learning_rate = learning_rate
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.95, staircase=True)


if use_different_dataset:
    # Local
    # train_dataset = "C:/Users/Florian/Desktop/dataset_3_apply_filter"
    # train_dataset = "C:/Users/Florian/Desktop/dataset_3_apply_filter_crowd"
    # train_dataset = "C:/Users/Florian/Desktop/dataset_3_apply_filter_crowd_min"

    # Server
    # train_dataset = "../datasets/dataset_3_apply_filter"
    # train_dataset = "../datasets/dataset_3_apply_filter_crowd"
    train_dataset = "../datasets/dataset_3_apply_filter_crowd_min"
else:
    train_dataset = 'dataset_mmp'



