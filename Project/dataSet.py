import numpy as np
import random
from annotationRect import AnnotationRect
import geometry
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops

image_height = 320
image_width = 320

crop_factor = 0.2
augmentation_factor = 0.15

# returns images of shape [batch_size, 2, 320, 320, 3] and labels of shape [batch_size, f_map_rows, f_map_cols, len(scales), len(aspect_ratios)]
# [batch_size, 0] are raw images, [batch_size, 1] are ground truth annotated images
def create(path, anchor_grid, iou):
    
    dataset = tf.data.Dataset.list_files(path+'/*.jpg')

    def create_label_array(lines):
        rects = []
        lines_decoded = str(lines.decode("utf-8")).split("\n")
        for line in lines_decoded:
            if(line):
                tokens = line.split(' ')
                annotation_rect = ([[float(tokens[1]) / image_height, float(tokens[0]) / image_width, float(tokens[3]) / image_height, float(tokens[2]) / image_width]])
                rects.append(annotation_rect)

        return np.asarray(rects, dtype=np.float32)


    def get_bounding_box_images(label_array):
        bounding_box_images = tf.zeros([len(label_array), image_height, image_width, 1], dtype=tf.float32)
        bounding_box_images = tf.image.draw_bounding_boxes(bounding_box_images, label_array)
        return bounding_box_images


    def parse_image(file_name):
        image = tf.io.read_file(file_name)
        image = tf.image.decode_jpeg(image)
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        image = tf.pad(image, [[0, image_height - h], [0, image_width - w], [0, 0]], mode='CONSTANT', constant_values=0)
        # Normalize
        return tf.cast(image, tf.float32) / 127.5 - 1


    def bmp_to_annotation_rect(img):
        img = (img > 0)
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        h_min, h_max = np.argmax(cols), img.shape[1] - 1 - np.argmax(np.flipud(cols))
        w_min, w_max = np.argmax(rows), img.shape[0] - 1 - np.argmax(np.flipud(rows))
        return AnnotationRect(h_min, w_min, h_max, w_max)


    def bounding_box_images_to_label(bounding_box_images):
        gt_ar_boxes = []
        gt_boxes = []
        for bounding_box_image in bounding_box_images:
            gt_box = bmp_to_annotation_rect(bounding_box_image)
            gt_ar_boxes.append(gt_box)
            gt_boxes.append([float(gt_box.y1) / image_height, float(gt_box.x1) / image_width, float(gt_box.y2) / image_height, float(gt_box.x2) / image_width])

        max_overlaps = geometry.anchor_max_gt_overlaps(anchor_grid, gt_ar_boxes)
        iou_boxes = (max_overlaps > iou).astype(np.int32)

        return iou_boxes, np.array(gt_boxes, dtype=np.float32)


    def random_rotate(image, bb_images):
        random_angle = tf.random.uniform([1], minval = -np.pi, maxval = np.pi)
        random_rotate_matrix = tf.contrib.image.angles_to_projective_transforms(random_angle, tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32))
        rotated_image = tf.contrib.image.transform(image, random_rotate_matrix)
        rotated_bb_images = tf.contrib.image.transform(bb_images, random_rotate_matrix)
        return rotated_image, rotated_bb_images


    def random_crop(image, bb_images):
        image_dim = tf.constant([image_height, image_width], dtype=tf.int32)
        
        rand_hw_start = tf.random.uniform([2], minval=0, maxval=crop_factor, dtype=tf.float32)
        rand_hw_size = tf.random.uniform([2], minval=(1.0 - crop_factor), maxval=1.0, dtype=tf.float32) - rand_hw_start

        rand_hw_start = tf.cast(rand_hw_start * tf.cast(image_dim, dtype=tf.float32), dtype=tf.int32)
        rand_hw_size = tf.cast(rand_hw_size * tf.cast(image_dim, dtype=tf.float32), dtype=tf.int32)

        rand_image_start = tf.concat([rand_hw_start, [tf.constant(0)]], 0)
        rand_image_size = tf.concat([rand_hw_size, [tf.shape(image)[-1]]], 0)
        image = tf.slice(image, rand_image_start, rand_image_size)
        image = tf.image.resize_images(image, image_dim)

        rand_bb_image_start = tf.concat([[tf.constant(0)], rand_hw_start], 0)
        rand_bb_image_size = tf.concat([[tf.shape(bb_images)[0]], rand_hw_size], 0)

        rand_bb_image_start = tf.concat([rand_bb_image_start, [tf.constant(0)]], 0)
        rand_bb_image_size = tf.concat([rand_bb_image_size, [tf.shape(bb_images)[-1]]], 0)

        bb_images = tf.slice(bb_images, rand_bb_image_start, rand_bb_image_size)
        bb_images = tf.image.resize_images(bb_images, image_dim)
        
        return image, bb_images

    def random_flip(image, bb_images):
        flip_vertically = random.choice([True, False])
        flip_horizontally = random.choice([True, False])

        if flip_vertically == True:
            image = tf.image.flip_left_right(image)
            bb_images = tf.image.flip_left_right(bb_images)
        
        if flip_horizontally == True:
            image = tf.image.flip_up_down(image)
            bb_images = tf.image.flip_up_down(bb_images)

        return image, bb_images        


    def random_quality(image, bb_images):
        image = tf.image.random_jpeg_quality(image, 50, 100)
        return image, bb_images


    def random_color(image, bb_images):
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        return image, bb_images


    def random_image_augmentation(image, bb_images):
        augmentations = [random_rotate,
                         random_flip,
                         random_quality,
                         random_color,
                         random_crop]

        def no_augment(arg1, arg2):
            return arg1, arg2
        
        for augmentation in augmentations:
            image, bb_images = tf.cond(tf.random_uniform([], 0.0, 1.0) < augmentation_factor, lambda: augmentation(image, bb_images), lambda: no_augment(image, bb_images))

        return image, bb_images


    def get_image_label_and_gt(file_name):
        annotation_file = tf.io.read_file(tf.strings.split(file_name, '.jpg', result_type='RaggedTensor')[0] + '.gt_data.txt')
        label_array = tf.py_func(create_label_array, [annotation_file], tf.float32)

        image = parse_image(file_name)
        bb_images = get_bounding_box_images(label_array)

        image, bb_images = random_image_augmentation(image, bb_images)

        iou_boxes, gt_boxes = tf.py_func(bounding_box_images_to_label, [bb_images], [tf.int32, tf.float32])

        image_annotated_gt = tf.squeeze(tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.expand_dims(gt_boxes, 0)))

        return tf.stack([image, image_annotated_gt]), iou_boxes


    dataset = dataset.map(get_image_label_and_gt).cache()

    return dataset