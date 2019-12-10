from PIL import ImageDraw, Image
import numpy as np
import os
import random
from annotationRect import AnnotationRect
import geometry
from scipy.special import softmax

folder_offset = 'dataset_mmp/'

image_height = 320
image_width = 320

crop_scales = list(np.arange(0.8, 1.0, 0.1))
crop_boxes = np.zeros((len(crop_scales), 4))

for i, scale in enumerate(crop_scales):
    x1 = y1 = 0.5 - (0.5 * scale)
    x2 = y2 = 0.5 + (0.5 * scale)
    crop_boxes[i] = [x1, y1, x2, y2]


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

        #boxes = np.zeros(anchor_grid.shape[:-1], dtype=np.int64)
        #boxes[indices] = 1
        #boxes = np.expand_dims(boxes, axis=-1)
        labels.append(labelgrid)
    return images, labels, gt_rects


def convert_to_annotation_rects_output(anchor_grid, output):
    calc_softmax = softmax(output, axis=-1)
    foreground = np.delete(calc_softmax, [0], axis=-1)
    filtered_indices = np.where(foreground > 0.7)
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



# returns images of shape [batch_size, 320, 320, 3] and labels of shape [batch_size, f_map_rows, f_map_cols, len(scales), len(aspect_ratios)]
# if include_annotated_gt_image is set to True images has shape images of shape [batch_size, 2, 320, 320, 3], one orignial and one gt annotaded image
def get_batch(batch_size, anchor_grid, iou, include_annotated_gt_image=False):
    
    dataset = tf.data.Dataset.list_files('./dataset_mmp/train/*.jpg')

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


    # todo
    def random_crop(image, bb_images):
        rand_crop_index = tf.random_uniform(shape=[], minval=0, maxval=len(crop_scales), dtype=tf.int32)
        image_crops = tf.image.crop_and_resize([image], boxes=crop_boxes, box_ind=np.zeros(len(crop_scales)), crop_size=[image_height, image_width])
        bb_images_crops = tf.image.crop_and_resize(bb_images, boxes=crop_boxes, box_ind=np.zeros(len(crop_scales)), crop_size=[image_height, image_width])
        return image_crops[rand_crop_index], bb_images_crops[rand_crop_index]


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
                         random_color]

        def no_augment(arg1, arg2):
            return arg1, arg2
        
        for augmentation in augmentations:
            image, bb_images = tf.cond(tf.random_uniform([], 0.0, 1.0) > 0.75, lambda: augmentation(image, bb_images), lambda: no_augment(image, bb_images))

        return image, bb_images



    def get_image_label_and_gt(file_name):
        annotation_file = tf.io.read_file(tf.strings.split(file_name, '.jpg', result_type='RaggedTensor')[0] + '.gt_data.txt')
        label_array = tf.py_func(create_label_array, [annotation_file], tf.float32)

        image = parse_image(file_name)
        bb_images = get_bounding_box_images(label_array)

        image, bb_images = random_image_augmentation(image, bb_images)

        iou_boxes, gt_boxes = tf.py_func(bounding_box_images_to_label, [bb_images], [tf.int32, tf.float32])

        image_annotated_gt = tf.squeeze(tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), tf.expand_dims(gt_boxes, 0)))

        if include_annotated_gt_image == True:
            return tf.stack([image, image_annotated_gt]), iou_boxes

        return image, iou_boxes


    dataset = dataset.map(get_image_label_and_gt)

    batch = dataset.batch(batch_size)

    iterator = batch.make_one_shot_iterator()

    images, labels = iterator.get_next()

    return images, labels
