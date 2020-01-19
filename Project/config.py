import tensorflow as tf

image_width = 320
image_height = 320
f_map_rows = 10
f_map_cols = 10
scale_factor = 32.0
scales = [50, 80, 100, 150]
aspect_ratios = [1.0, 1.5, 2.0]
batch_size = 30
iou = 0.5
learning_rate = 0.001
iterations = 1000
negative_example_factor = 10
output_image_size = (1280, 1280)