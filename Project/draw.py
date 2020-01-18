import dataUtil
from PIL import ImageDraw, Image
import config
import os
import numpy as np
import evaluation
from annotationRect import AnnotationRect


def __convert_to_annotation_rects_label(anchor_grid, labels):
    indices = np.where(labels == 1)[:4]
    max_boxes = anchor_grid[indices]
    annotated_boxes = [AnnotationRect(*max_boxes[i]) for i in range(len(max_boxes))]
    return annotated_boxes


def __draw_bounding_boxes(image, annotation_rects, color):
    for rect in annotation_rects:
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            xy=[rect.x1, rect.y1, rect.x2, rect.y2],
            outline=color
        )
    return image


def __draw_labels(image, anchor_grid, labels, color):
    annotation_rects = __convert_to_annotation_rects_label(anchor_grid, labels)
    for rect in annotation_rects:
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            xy=[rect.x1, rect.y1, rect.x2, rect.y2],
            outline=color
        )
    return image


def draw_images(num_images, images, output, labels, gts, adjusted_anchor_grid, original_anchor_grid, path):

    path = path + 'test_images/'
    if not os.path.exists(path):
        os.makedirs(path)

    for k in range(num_images):

        '''
        Normal
        '''
        boxes_dict = evaluation.create_boxes_dict(output[k], original_anchor_grid, 0.5)
        nms = evaluation.non_maximum_suppression(boxes_dict)
        boxes = list(nms.keys())

        image = Image.fromarray(((images[k] + 1) * 127.5).astype(np.uint8), 'RGB')
        __draw_bounding_boxes(image, gts[k], (255, 0, 0))
        dataUtil.draw_labels(image, original_anchor_grid, labels[k], (0, 255, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_labels.jpg'.format(k))

        __draw_bounding_boxes(image, boxes, (0, 0, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_estimates.jpg'.format(k))

        '''
        Bounding box regression
        '''
        if config.use_bounding_box_regression:
            boxes_dict_bbr = evaluation.create_boxes_dict(output[k], adjusted_anchor_grid[k], 0.5)
            nms_bbr = evaluation.non_maximum_suppression(boxes_dict_bbr)
            boxes_bbr = list(nms_bbr.keys())

            image_adjusted = Image.fromarray(((images[k] + 1) * 127.5).astype(np.uint8), 'RGB')
            __draw_bounding_boxes(image_adjusted, gts[k], (255, 0, 0))
            dataUtil.draw_labels(image_adjusted, adjusted_anchor_grid[k], labels[k], (0, 255, 255))
            image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_labels_bbr.jpg'.format(k))

            __draw_bounding_boxes(image_adjusted, boxes_bbr, (0, 0, 255))
            image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_estimates_bbr.jpg'.format(k))
