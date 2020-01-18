from PIL import Image
import numpy as np
import tensorflow as tf
import os
from annotationRect import AnnotationRect
import geometry
import config

folder_offset = 'dataset_mmp/'


def __convert_file_annotation_rect(location):
    file = open(location, 'r')
    if file.mode == 'r':
        lines = file.readlines()
        rects = []
        for l in lines:
            tokens = l.split(' ')
            rects.append(AnnotationRect(int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])))
        return rects


def __get_dict_from_folder(folder):
    result_dict = {}
    for file in os.listdir(folder_offset + folder):
        if file.endswith(".txt"):
            location = folder_offset + folder + '/' + file
            key = folder_offset + folder + '/' + file.split('.', 1)[0] + '.jpg'
            result_dict[key] = __convert_file_annotation_rect(location)
    return result_dict


def get_validation_data(package_size, anchor_grid):
    items = __get_dict_from_folder('test')
    images = []
    labels = []
    gt_rects = []
    image_paths = []
    counter = 0
    result = []
    for path in items:

        # Path
        image_paths.append(path)
        gt_annotation_rects = items.get(path)
        gt_rects.append(gt_annotation_rects)

        # Images
        img = np.array(Image.open(path))
        h, w = img.shape[:2]
        img_pad = np.pad(img, pad_width=((0, 320 - h), (0, 320 - w), (0, 0)), mode='constant', constant_values=0)
        img_norm = img_pad.astype(np.float32) / 127.5 - 1
        images.append(img_norm)

        # Labels
        max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid, gt_annotation_rects)
        label_grid = (max_overlaps > config.iou).astype(np.int32)
        labels.append(label_grid)
        counter += 1
        if counter % package_size is 0:
            result.append((np.asarray(images), np.asarray(labels), np.asarray(gt_rects), np.asarray(image_paths)))
            images = []
            labels = []
            gt_rects = []
            image_paths = []
    result.append((np.asarray(images), np.asarray(labels), np.asarray(gt_rects), np.asarray(image_paths)))
    return result


def calculate_adjusted_anchor_grid(anchor_grid, adjustments):
    num_batch_size = tf.shape(adjustments)[0]
    ag_batched = tf.cast(tf.tile(tf.expand_dims(anchor_grid, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float32)

    # Inverted regression targets
    ag_sizes = ag_batched[..., 2:4] - ag_batched[..., 0:2]
    lower_adjusted = adjustments[..., 0:2] * ag_sizes + ag_batched[..., 0:2]
    sizes_adjusted = tf.math.exp(adjustments[..., 2:4]) * ag_sizes
    upper_adjusted = lower_adjusted + sizes_adjusted

    ag_adjusted = tf.concat([lower_adjusted, upper_adjusted], -1)
    return ag_adjusted