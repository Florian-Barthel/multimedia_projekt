import os
import numpy as np
from PIL import Image
import data

image_path = 'test_images'
max_drawn_images = 10
fg_threshold = 0.7

def draw_images(test_images, test_labels, output, anchor_grid, gt_annotation_rects, nms_boxes, num_test_images):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # Limits number of drawn images to 10 by default
    num_test_images = min(num_test_images, max_drawn_images)
    # Draw normal boxes
    for i in range(num_test_images):
        img = Image.fromarray(((test_images[i] + 1) * 128).astype(np.uint8), 'RGB')
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=data.convert_to_annotation_rects_label(anchor_grid,
                                                                                         test_labels[i]),
                                 color=(255, 100, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=data.convert_to_annotation_rects_output(anchor_grid, output[i]),
                                 color=(100, 255, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=gt_annotation_rects[i],
                                 color=(100, 100, 255))
        img.save(image_path + '/max_overlap_boxes_{}.jpg'.format(i))
    # Draw NMS boxes
    for i in range(num_test_images):
        for b in list(nms_boxes[i]):
            if nms_boxes[i][b] < fg_threshold:
                nms_boxes[i].pop(b)
        img = Image.fromarray(((test_images[i] + 1) * 128).astype(np.uint8), 'RGB')
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=data.convert_to_annotation_rects_label(anchor_grid,
                                                                                         test_labels[i]),
                                 color=(255, 100, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=list(nms_boxes[i].keys()),
                                 color=(100, 255, 100))
        data.draw_bounding_boxes(image=img,
                                 annotation_rects=gt_annotation_rects[i],
                                 color=(100, 100, 255))
        img.save(image_path + '/nms_overlap_boxes_{}.jpg'.format(i))
