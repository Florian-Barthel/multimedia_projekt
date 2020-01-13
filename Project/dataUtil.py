from PIL import ImageDraw, Image
import numpy as np
import tensorflow as tf
import os
import random

import fileUtil
from annotationRect import AnnotationRect
import geometry
import config
from config import image_height, image_width
from scipy.special import softmax

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


def get_dict_from_folder(folder):
    result_dict = {}
    for file in os.listdir(folder_offset + folder):
        if file.endswith(".txt"):
            location = folder_offset + folder + '/' + file
            key = folder_offset + folder + '/' + file.split('.', 1)[0] + '.jpg'
            result_dict[key] = __convert_file_annotation_rect(location)
    return result_dict


def draw_bounding_boxes(image, annotation_rects, color):
    for rect in annotation_rects:
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            xy=[rect.x1, rect.y1, rect.x2, rect.y2],
            outline=color
        )
    return image

def calculate_overlap_boxes_tensor(labels_tensor, anchor_grid_tensor):

    def py_get_overlap_boxes(labels):
        ar_boxes = []
        
        for i in range(labels.shape[0]):
            ar_batch_boxes = []

            for label in labels[i]:
                if(np.isnan(np.sum(label)) == False):
                    ar_batch_boxes.append(AnnotationRect(int(label[1] * image_height), int(label[0] * image_width), int(label[3] * image_height), int(label[2] * image_width)))

            max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid_tensor, ar_batch_boxes)
            iou_boxes = (max_overlaps > config.iou).astype(np.int32)
            ar_boxes.append(iou_boxes)

        return np.array(ar_boxes, dtype=np.int32)

    return tf.py_func(py_get_overlap_boxes, [labels_tensor], tf.int32)

def make_random_batch(batch_size, anchor_grid):
    items = get_dict_from_folder('train')
    images = []
    labels = []
    gt_rects = []
    image_paths = [None] * batch_size
    for i in range(batch_size):
        image_paths[i], gt_annotation_rects = random.choice(list(items.items()))
        gt_rects.append(gt_annotation_rects)

        img = np.array(Image.open(image_paths[i]))
        h, w = img.shape[:2]
        img_pad = np.pad(img, pad_width=((0, 320 - h), (0, 320 - w), (0, 0)), mode='constant', constant_values=0)
        img_norm = img_pad.astype(np.float32) / 127.5 - 1
        images.append(img_norm)

        max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid, gt_annotation_rects)
        label_grid = (max_overlaps > config.iou).astype(np.int32)
        labels.append(label_grid)
    return images, labels, gt_rects, image_paths


def get_validation_data(package_size, anchor_grid):
    items = get_dict_from_folder('test')
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


def convert_to_annotation_rects_output(anchor_grid, output, confidence=0.7):
    calc_softmax = softmax(output, axis=-1)
    foreground = np.delete(calc_softmax, [0], axis=-1)
    indices = np.where(foreground > confidence)[:4]
    max_boxes = anchor_grid[indices]
    return [AnnotationRect(*max_boxes[i]) for i in range(len(max_boxes))]


def convert_to_annotation_rects_label(anchor_grid, labels):
    indices = np.where(labels == 1)[:4]
    max_boxes = anchor_grid[indices]
    annotated_boxes = [AnnotationRect(*max_boxes[i]) for i in range(len(max_boxes))]
    return annotated_boxes


def calculate_adjusted_anchor_grid(ag, adjustments):
    
    num_batch_size = tf.shape(adjustments)[0]
    ag_batched = tf.cast(tf.tile(tf.expand_dims(ag, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float32)

    # Inverted regression targets
    ag_sizes = ag_batched[..., 2:4] - ag_batched[..., 0:2]
    lower_adjusted = adjustments[..., 0:2] * ag_sizes + ag_batched[..., 0:2]
    sizes_adjusted = tf.math.exp(adjustments[..., 2:4]) * ag_sizes
    upper_adjusted = lower_adjusted + sizes_adjusted

    ag_adjusted = tf.concat([lower_adjusted, upper_adjusted], -1)
    return ag_adjusted