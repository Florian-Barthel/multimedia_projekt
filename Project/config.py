import tensorflow as tf
from datetime import datetime

use_decaying_learning_rate = True
use_augmentation = True
use_adam_optimizer = True
use_hard_negative_mining = True
use_bounding_box_regression = True

image_width = 320
image_height = 320
output_image_size = (1280, 1280)

f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [80, 120, 150]
aspect_ratios = [1.0, 1.5, 2.0]

batch_size = 30
iterations = 40000
validation_interval = 500

iou = 0.5
negative_example_factor = 9

detection_foreground_threshold = 0.2
nms_threshold = 0.3

# Local
train_dataset = "C:/Users/Florian/Desktop/dataset_3_apply_filter_crowd_min"

# Server
# train_dataset = "../datasets/dataset_3_apply_filter_crowd_min"

# TensorBoard logs saved in ./logs/dd-MM-yyyy_HH-mm-ss
current_time = datetime.now()
logs_directory = './logs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S')
detection_directory = 'runs/' + current_time.strftime('%d-%m-%Y_%H-%M-%S') + '/'
validation_directory = 'dataset_mmp'

if use_decaying_learning_rate:
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0001
    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.95, staircase=True)
else:
    learning_rate = 0.0001
