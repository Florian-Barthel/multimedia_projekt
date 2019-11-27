from PIL import ImageDraw, Image
import numpy as np
import os
import random
from annotationRect import AnnotationRect
import anchorgrid
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
    for _ in range(batch_size):
        key, value = random.choice(list(items.items()))
        # img = np.array(Image.open(key))
        img = np.array(Image.open(key).resize((224, 224))).astype(np.float) / 128 - 1
        images.append(img)

        max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid, value)
        indices = np.where(max_overlaps > iou)
        boxes = np.zeros(anchor_grid.shape[:-1], dtype=np.int64)
        boxes[indices] = 1
        boxes = np.expand_dims(boxes, axis=-1)
        labels.append(boxes)
    return images, labels


# my_anchor_grid = anchorgrid.anchor_grid(fmap_rows=20,
#                                         fmap_cols=20,
#                                         scale_factor=16.0,
#                                         scales=[70, 100, 140, 200],
#                                         aspect_ratios=[0.5, 1.0, 2.0])
#
# (batch_images, batch_labels) = make_random_batch(5, my_anchor_grid, 0.5)
# print()
