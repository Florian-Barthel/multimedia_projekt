from PIL import ImageDraw, Image
import numpy as np
import os
import random
from annotationRect import AnnotationRect
import geometry
from scipy.special import softmax
import tensorflow as tf
from dataSet import image_height, image_width

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


def calculate_overlap_boxes_tensor(labels_tensor, anchor_grid_tensor, iou):

    def py_get_overlap_boxes(labels):
        ar_boxes = []
        
        for i in range(labels.shape[0]):
            ar_batch_boxes = []

            for label in labels[i]:
                if(np.isnan(np.sum(label)) == False):
                    ar_batch_boxes.append(AnnotationRect(int(label[1] * image_height), int(label[0] * image_width), int(label[3] * image_height), int(label[2] * image_width)))

            max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid_tensor, ar_batch_boxes)
            iou_boxes = (max_overlaps > iou).astype(np.int32)
            ar_boxes.append(iou_boxes)

        return np.array(ar_boxes, dtype=np.int32)

    return tf.py_func(py_get_overlap_boxes, [labels_tensor], tf.int32)


def convert_to_annotation_rects_output(anchor_grid, output):
    calc_softmax = softmax(output, axis=-1)
    foreground = np.delete(calc_softmax, [0], axis=-1)
    filtered_indices = np.where(foreground > 0.95)
    filtered_indices = np.where(foreground > 0.93)
    remove_last = filtered_indices[:4]
    max_boxes = anchor_grid[remove_last]
    return [AnnotationRect(*max_boxes[i]) for i in range(max_boxes.shape[0])]


def convert_to_annotation_rects_label(anchor_grid, labels):
    filtered_indices = np.where(labels == 1)
    remove_last = filtered_indices[:4]
    max_boxes = anchor_grid[remove_last]

    # * = tuple unpacking
    annotated_boxes = [AnnotationRect(*max_boxes[i]) for i in range(max_boxes.shape[0])]
    return annotated_boxes