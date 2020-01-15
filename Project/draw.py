from tqdm import tqdm
import dataUtil
from PIL import Image
import config
import os
import numpy as np


def draw_images(num_images, images, output, labels, gts, adjusted_anchor_grid, original_anchor_grid, path):

    path = path + 'test_images/'
    if not os.path.exists(path):
        os.makedirs(path)

    for k in range(num_images):
        image = Image.fromarray(((images[k] + 1) * 127.5).astype(np.uint8), 'RGB')
        dataUtil.draw_bounding_boxes(image, gts[k], (255, 0, 0))
        # image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_gts.jpg'.format(k))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_label(original_anchor_grid, labels[k]),
                                     (0, 255, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_labels.jpg'.format(k))

        dataUtil.draw_bounding_boxes(image, dataUtil.convert_to_annotation_rects_output(original_anchor_grid, output[k]),
                                     (0, 0, 255))
        image.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_estimates.jpg'.format(k))

        image_adjusted = Image.fromarray(((images[k] + 1) * 127.5).astype(np.uint8), 'RGB')
        dataUtil.draw_bounding_boxes(image_adjusted, gts[k], (255, 0, 0))
        dataUtil.draw_bounding_boxes(image_adjusted,
                                     dataUtil.convert_to_annotation_rects_label(adjusted_anchor_grid[k], labels[k]),
                                     (0, 255, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(path + '{}_labels_bbr.jpg'.format(k))

        dataUtil.draw_bounding_boxes(image_adjusted,
                                     dataUtil.convert_to_annotation_rects_output(adjusted_anchor_grid[k], output[k]),
                                     (0, 0, 255))
        image_adjusted.resize(config.output_image_size, Image.ANTIALIAS).save(
            path + '{}_estimates_bbr.jpg'.format(k))
