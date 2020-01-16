from tqdm import tqdm
import dataUtil
from PIL import Image
import config
import os
import numpy as np
import evaluation


def draw_images(num_images, images, output, labels, gts, adjusted_anchor_grid, original_anchor_grid, path):

    path = path + 'test_images/'
    if not os.path.exists(path):
        os.makedirs(path)

    for k in range(num_images):

        '''
        Normal
        '''
        boxes_dict = evaluation.create_boxes_dict(output[k], original_anchor_grid)
        nms = evaluation.non_maximum_suppression(boxes_dict)
        boxes = list(nms.keys())

        image = Image.fromarray(((images[k] + 1) * 127.5).astype(np.uint8), 'RGB')
        dataUtil.draw_bounding_boxes(image, gts[k], (255, 0, 0))
        dataUtil.draw_labels(image, original_anchor_grid, labels[k], (0, 255, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_labels.jpg'.format(k))

        dataUtil.draw_bounding_boxes(image, boxes, (0, 0, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_estimates.jpg'.format(k))

        '''
        Bounding box regression
        '''
        boxes_dict_bbr = evaluation.create_boxes_dict(output[k], adjusted_anchor_grid[k])
        nms_bbr = evaluation.non_maximum_suppression(boxes_dict_bbr)
        boxes_bbr = list(nms_bbr.keys())

        image_adjusted = Image.fromarray(((images[k] + 1) * 127.5).astype(np.uint8), 'RGB')
        dataUtil.draw_bounding_boxes(image_adjusted, gts[k], (255, 0, 0))
        dataUtil.draw_labels(image_adjusted, adjusted_anchor_grid[k], labels[k], (0, 255, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_labels_bbr.jpg'.format(k))

        dataUtil.draw_bounding_boxes(image_adjusted, boxes_bbr, (0, 0, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_estimates_bbr.jpg'.format(k))
