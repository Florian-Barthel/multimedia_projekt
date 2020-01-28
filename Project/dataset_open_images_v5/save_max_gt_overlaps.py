import os
from annotationRect import AnnotationRect
import pickle
import geometry
import config
import numpy as np
from tqdm import tqdm

# dataset_directory = 'C:/Users/Florian/Desktop/dataset_2_crowd_min'
dataset_directory = '../../datasets/dataset_2_crowd_min_plus_mmp_dataset_train'
# dataset_directory = '../dataset_mmp/train'
anchor_grid_configuration = '[60, 90, 120, 150, 250]_[0.5, 0.75, 1.0, 1.5, 2.0]_2' + dataset_directory.split('/')[-1]
target_directory = '../max_gt_overlaps_objects/'

if not os.path.exists(target_directory):
    os.makedirs(target_directory)


def __convert_file_annotation_rect(location):
    file = open(location, 'r')
    if file.mode == 'r':
        lines = file.readlines()
        rects = []
        for l in lines:
            tokens = l.split(' ')
            rects.append([int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])])
        return rects


def get_dict_from_folder(folder):
    result_dict = {}
    for file in tqdm(os.listdir(folder)):
        if file.endswith(".txt"):
            location = folder + '/' + file
            key = file.split('.', 1)[0] + '.jpg'
            result_dict[key] = __convert_file_annotation_rect(location)
    return result_dict


def get_label_grid(gts):
    rects = []
    for i in range(len(gts)):
        rects.append(AnnotationRect(int(gts[i][0]), int(gts[i][1]), int(gts[i][2]), int(gts[i][3])))
    max_overlaps = geometry.anchor_max_gt_overlaps(config.anchor_grid, rects)
    label_grid = (max_overlaps > config.iou).astype(np.int32)
    return label_grid


data = get_dict_from_folder(dataset_directory)

max_gt_overlaps_dict = {}

for image_name, gts in tqdm(data.items()):
    max_gt_overlaps_dict[image_name] = get_label_grid(gts)


with open(target_directory + anchor_grid_configuration + '.pkl', 'wb') as handle:
    pickle.dump(max_gt_overlaps_dict, handle)


with open(target_directory + anchor_grid_configuration + '.pkl', 'rb') as handle:
    stored_dict = pickle.load(handle)

print(len(max_gt_overlaps_dict), len(stored_dict))

print(max_gt_overlaps_dict.keys() - stored_dict.keys())
