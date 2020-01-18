import numpy as np
from annotationRect import AnnotationRect
import geometry
import tensorflow as tf
import config
from config import image_height, image_width
import pickle

with open('max_gt_overlaps_objects/' + '[80,100,150]_[0.5,1.0,2.0]_dataset_3_apply_filter_crowd_min.pkl',
          'rb') as handle:
    max_gt_overlap_dict = pickle.load(handle)


# returns images of shape [batch_size, 320, 320, 3]

def get_gt_array(lines):
    rects = []
    lines_decoded = str(lines.decode("utf-8")).split("\n")
    for line in lines_decoded:
        if line:
            tokens = line.split(' ')
            annotation_rect = ([tokens[0], tokens[1], tokens[2], tokens[3]])
            rects.append(annotation_rect)
    return np.asarray(rects, dtype=np.float32)


def get_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    h_pad = image_height - h
    w_pad = image_width - w

    image = tf.pad(image, [[0, h_pad], [0, w_pad], [0, 0]], mode='CONSTANT', constant_values=0)
    # Normalize
    image = tf.cast(image, tf.float32) / 127.5 - 1

    return image


def get_label_grid(gts):
    rects = []
    for gt in gts:
        rects.append(AnnotationRect(int(gt[0]), int(gt[1]), int(gt[2]), int(gt[3])))
    max_overlaps = geometry.anchor_max_gt_overlaps(config.anchor_grid, rects)
    label_grid = (max_overlaps > config.iou).astype(np.int32)
    return label_grid


def get_label_grid_fast(image_name):
    image_name = image_name.decode('utf-8')
    image_name = image_name.replace('\\', '/')
    image_name = image_name.split('/')[-1]
    return max_gt_overlap_dict[str(image_name)]


def random_flip(image, gts):
    image = tf.image.flip_left_right(image)
    width = tf.constant([image_width, 0, image_width, 0], dtype=tf.float32)
    gts_flipped = width - gts
    return image, gts_flipped


def random_color(image, gts):
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    return image, gts


def random_image_augmentation(image, gt_array):
    augmentation_functions = [random_color,
                              random_flip]

    def no_augmentation(keep_image, keep_gt_array):
        return keep_image, keep_gt_array

    for augmentation in augmentation_functions:
        image, gt_array = tf.cond(tf.random_uniform([], 0.0, 1.0) < config.augmentation_probability,
                                  lambda: augmentation(image, gt_array), lambda: no_augmentation(image, gt_array))

    return image, gt_array


def get_image_label_gt(file_name):
    image_name = tf.strings.split(file_name, '.jpg', result_type='RaggedTensor')[0]
    annotation_file = tf.io.read_file(image_name + '.gt_data.txt')

    gt_array = tf.py_func(get_gt_array, [annotation_file], Tout=tf.float32)
    image = get_image(file_name)

    if config.use_augmentation:
        image, gt_array = random_image_augmentation(image, gt_array)

    # label_grid = tf.py_func(get_label_grid, [gt_array], Tout=tf.int32)
    label_grid = tf.py_func(get_label_grid_fast, [file_name],
                            Tout=tf.int32)

    return image, gt_array, label_grid


def create(path, batch_size):
    dataset = tf.data.Dataset.list_files(path + '/*.jpg')
    return dataset.map(get_image_label_gt).repeat().padded_batch(batch_size,
                                                                 padded_shapes=(
                                                                     [image_height, image_width, 3], [None, 4],
                                                                     [config.f_map_rows, config.f_map_cols,
                                                                      len(config.scales),
                                                                      len(config.aspect_ratios)]),
                                                                 padding_values=(0.0, 0.0, 0)).prefetch(
        tf.data.experimental.AUTOTUNE)
