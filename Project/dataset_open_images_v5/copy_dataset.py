from tqdm import tqdm
import shutil
import os

'''
Filter all boxes with greater width than height
'''

# src_folder = '../datasets/dataset_2_crowd_min_ratio'
# dest_folder = '../datasets/dataset_2_crowd_min_ratio_flipped'

src_folder = 'C:/Users/Florian/Desktop/dataset_2_crowd_min_ratio'
dest_folder = 'C:/Users/Florian/Desktop/dataset_2_crowd_min_ratio_flipped_union'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0
num_filtered = 0

for file in tqdm(os.listdir(src_folder)):
    if file.endswith(".txt"):
        src_txt = src_folder + '/' + file
        src_img = src_folder + '/' + file.split('.', 1)[0] + '.jpg'
        dest_txt = dest_folder + '/' + file
        dest_img = dest_folder + '/' + file.split('.', 1)[0] + '.jpg'

        shutil.copyfile(src_txt, dest_txt)
        shutil.copyfile(src_img, dest_img)
        num_kept += 1

print('kept: ' + str(num_kept))
print('removed: ' + str(num_filtered))
