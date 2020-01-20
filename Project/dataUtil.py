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


def calculate_adjusted_anchor_grid_working(anchor_grid, adjustments):
    num_batch_size = tf.shape(adjustments)[0]
    ag_batched = tf.cast(tf.tile(tf.expand_dims(anchor_grid, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float32)

    # Inverted regression targets
    ag_sizes = ag_batched[..., 2:4] - ag_batched[..., 0:2]
    lower_adjusted = adjustments[..., 0:2] * ag_sizes + ag_batched[..., 0:2]
    sizes_adjusted = tf.math.exp(adjustments[..., 2:4]) * ag_sizes
    upper_adjusted = lower_adjusted + sizes_adjusted

    ag_adjusted = tf.concat([lower_adjusted, upper_adjusted], -1)
    return ag_adjusted


def calculate_adjusted_anchor_grid(anchor_grid, adjustments):
    # TODO: Replace concat with stack
    num_batch_size = tf.shape(adjustments)[0]
    ag_batched = tf.cast(tf.tile(tf.expand_dims(anchor_grid, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float32)

    # Inverted regression targets
    # ag_sizes = tf.abs(ag_batched[..., 2:4] - ag_batched[..., 0:2])
    # lower_adjusted = adjustments[..., 0:2] * ag_sizes + ag_batched[..., 0:2]
    # sizes_adjusted = tf.math.exp(adjustments[..., 2:4]) * ag_sizes
    # upper_adjusted = lower_adjusted + sizes_adjusted

    # Invert
    width_anchor = tf.abs(ag_batched[..., 2] - ag_batched[..., 0])
    height_anchor = tf.abs(ag_batched[..., 2] - ag_batched[..., 0])
    tx = adjustments[..., 0] * width_anchor
    tx = tx + ag_batched[..., 0]
    ty = adjustments[..., 1] * height_anchor
    ty = ty + ag_batched[..., 1]

    sx = tf.exp(adjustments[..., 2])
    sx = sx * width_anchor

    sy = tf.exp(adjustments[..., 3])
    sy = sy * height_anchor

    tx = tf.expand_dims(tx, -1)
    ty = tf.expand_dims(ty, -1)
    sx = tf.expand_dims(sx, -1)
    sy = tf.expand_dims(sy, -1)

    inverted_adjustments = tf.concat([tx, ty, sx, sy], -1)

    width = ag_batched[..., 2] - ag_batched[..., 0]
    height = ag_batched[..., 3] - ag_batched[..., 1]
    new_width = width * inverted_adjustments[..., 2]
    new_height = height * inverted_adjustments[..., 3]

    x1 = ag_batched[..., 0] + inverted_adjustments[..., 0]
    y1 = ag_batched[..., 1] + inverted_adjustments[..., 1]

    x1 = tf.expand_dims(x1, -1)
    y1 = tf.expand_dims(y1, -1)

    # TODO: remove redundant concat
    offsets = tf.concat([x1, y1], axis=-1)

    x2 = offsets[..., 0] + new_width
    y2 = offsets[..., 1] + new_height

    x2 = tf.expand_dims(x2, -1)
    y2 = tf.expand_dims(y2, -1)

    offset_and_scales = tf.concat([x1, y1, x2, y2], axis=-1)

    return offset_and_scales

