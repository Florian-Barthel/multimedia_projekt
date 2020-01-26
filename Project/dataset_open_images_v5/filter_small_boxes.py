from tqdm import tqdm
import shutil
import os
from annotationRect import AnnotationRect

'''
Filters all images that have more than max_labels labels
'''
min_size = 1000


src_folder = 'C:/Users/Florian/Desktop/dataset_3_apply_filter_crowd'
dest_folder = 'C:/Users/Florian/Desktop/dataset_3_apply_filter_crowd_min'
remove_folder = 'C:/Users/Florian/Desktop/dataset_3_apply_filter_crowd_min_removed'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

if not os.path.exists(remove_folder):
    os.makedirs(remove_folder)

num_kept = 0
num_filtered = 0

for file in tqdm(os.listdir(src_folder)):
    if file.endswith(".txt"):
        src_txt = src_folder + '/' + file
        src_img = src_folder + '/' + file.split('.', 1)[0] + '.jpg'
        dest_txt = dest_folder + '/' + file
        dest_img = dest_folder + '/' + file.split('.', 1)[0] + '.jpg'
        dest_txt_remove = remove_folder + '/' + file
        dest_img_remove = remove_folder + '/' + file.split('.', 1)[0] + '.jpg'
        file = open(src_txt, 'r')
        good_sizes = True
        if file.mode == 'r':
            lines = file.readlines()
            for l in lines:
                tokens = l.split(' ')
                rect = AnnotationRect(int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3]))
                if not min_size < rect.area():
                    good_sizes = False

        if good_sizes:
            shutil.copyfile(src_txt, dest_txt)
            shutil.copyfile(src_img, dest_img)
            num_kept += 1
        else:
            shutil.copyfile(src_txt, dest_txt_remove)
            shutil.copyfile(src_img, dest_img_remove)
            num_filtered += 1

print('kept: ' + str(num_kept))
print('removed: ' + str(num_filtered))
