import numpy as np
import dataUtil
from PIL import Image
import os
from annotationRect import AnnotationRect

folder = 'C:/Users/Florian/Desktop/train_v4'


def __convert_file_annotation_rect(location):
    file = open(location, 'r')
    if file.mode == 'r':
        lines = file.readlines()
        rects = []
        for l in lines:
            tokens = l.split(' ')
            rects.append(AnnotationRect(int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])))
        return rects


def get_dict_from_folder(num):
    counter = 0
    result_dict = {}
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            location = folder + '/' + file
            key = folder + '/' + file.split('.', 1)[0] + '.jpg'
            result_dict[key] = __convert_file_annotation_rect(location)
        if counter >= num:
            return result_dict
        counter += 1
    return result_dict


def get_data():
    items = get_dict_from_folder(1000)
    images = []
    gt_rects = []
    for path in items:

        # Path
        gt_annotation = items.get(path)
        gt_rects.append(gt_annotation)

        # Images
        img = np.array(Image.open(path).convert('RGB'))
        h, w = img.shape[:2]
        img_pad = np.pad(img, pad_width=((0, 320 - h), (0, 320 - w), (0, 0)), mode='constant', constant_values=0)
        img_norm = img_pad.astype(np.float32) / 127.5 - 1
        images.append(img_norm)
    return np.asarray(images), np.asarray(gt_rects)


def test_labels(num):
    data = get_data()
    (test_images, gt_annotation_rects) = data

    for i in range(100):
        image = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        dataUtil.draw_bounding_boxes(image=image,
                                     annotation_rects=gt_annotation_rects[i],
                                     color=(100, 100, 255))
        if not os.path.exists('test_images'):
            os.makedirs('test_images')
        image.save('test_images/test_image_{}.jpg'.format(i))


'''
Number of Images to test
'''
test_labels(100)
