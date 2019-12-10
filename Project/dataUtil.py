from PIL import ImageDraw, Image
import numpy as np
import os
from annotationRect import AnnotationRect

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