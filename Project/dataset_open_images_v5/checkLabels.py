import numpy as np
from PIL import Image
import os
import draw
from annotationRect import AnnotationRect

folder = 'C:/Users/Florian/Desktop/dataset_1'


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


def get_data(num):
    items = get_dict_from_folder(num)
    images = []
    gt_rects = []
    paths = []
    for path in items:
        paths.append(path)
        # Path
        gt_annotation = items.get(path)
        gt_rects.append(gt_annotation)

        # Images
        img = np.array(Image.open(path).convert('RGB'))
        h, w = img.shape[:2]
        img_pad = np.pad(img, pad_width=((0, 320 - h), (0, 320 - w), (0, 0)), mode='constant', constant_values=0)
        img_norm = img_pad.astype(np.float32) / 127.5 - 1
        images.append(img_norm)
    return np.asarray(images), np.asarray(gt_rects), paths


def test_labels(num):
    data = get_data(num * 2)
    (test_images, gt_annotation_rects, paths) = data

    for i in range(num):
        image = Image.fromarray(((test_images[i] + 1) * 127.5).astype(np.uint8), 'RGB')
        draw.draw_bounding_boxes(image=image,
                                 annotation_rects=gt_annotation_rects[i],
                                 color=(100, 100, 255))
        image.save('test_images/' + str(i) + '_' + paths[i].split('/')[-1].split('.')[0] + '_test_image.jpg')


'''
Number of Images to test
'''
test_dir = 'test_images'

if not os.path.exists(test_dir):
    os.makedirs(test_dir)
test_labels(100)
