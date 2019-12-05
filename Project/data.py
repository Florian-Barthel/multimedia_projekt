from PIL import ImageDraw, Image
import numpy as np
import os
import random
from annotationRect import AnnotationRect
import geometry

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
            outline=color,
            width=3
        )
    return image


def make_random_batch(batch_size, anchor_grid, iou):
    items = get_dict_from_folder('train')
    images = []
    labels = []
    gt_rects = []
    for _ in range(batch_size):
        image_path, gt_annotation_rects = random.choice(list(items.items()))
        gt_rects.append(gt_annotation_rects)

        img = np.array(Image.open(image_path))
        h, w = img.shape[:2]
        img_pad = np.pad(img, pad_width=((0, 320 - h), (0, 320 - w), (0, 0)), mode='constant', constant_values=0)
        img_norm = img_pad.astype(np.float) / 127.5 - 1
        images.append(img_norm)

        max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid, gt_annotation_rects)
        labelgrid = (max_overlaps > iou).astype(np.int32)
        labels.append(labelgrid)
    return images, labels, gt_rects


def convert_to_annotation_rects_output(anchor_grid, output):
    remove_dimension = np.argmax(output, axis=-1)
    filtered_indices = np.where(remove_dimension == 1)
    max_boxes = anchor_grid[filtered_indices]

    # * = tuple unpacking
    annotated_boxes = [AnnotationRect(*max_boxes[i]) for i in range(max_boxes.shape[0])]
    return annotated_boxes


def convert_to_annotation_rects_label(anchor_grid, labels):
    filtered_indices = np.where(labels == 1)
    remove_last = filtered_indices[:4]
    max_boxes = anchor_grid[remove_last]

    # * = tuple unpacking
    annotated_boxes = [AnnotationRect(*max_boxes[i]) for i in range(max_boxes.shape[0])]
    return annotated_boxes
