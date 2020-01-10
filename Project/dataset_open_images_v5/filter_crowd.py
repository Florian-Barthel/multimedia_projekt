from tqdm import tqdm
import shutil
import os

'''
Filters all images that have more than max_labels labels
'''
max_labels = 5

src_folder = 'C:/Users/Florian/Desktop/dataset_2'
dest_folder = 'C:/Users/Florian/Desktop/dataset_2_crowd'

num_kept = 0
num_filtered = 0

for file in tqdm(os.listdir(src_folder)):
    if file.endswith(".txt"):
        src_txt = src_folder + '/' + file
        src_img = src_folder + '/' + file.split('.', 1)[0] + '.jpg'
        dest_txt = dest_folder + '/' + file
        dest_img = dest_folder + '/' + file.split('.', 1)[0] + '.jpg'
        file = open(src_txt, 'r')
        counter = 0
        if file.mode == 'r':
            lines = file.readlines()
            for l in lines:
                counter += 1

        # if the image has more than max_labels labels it doesnt get copied
        if 0 < counter <= max_labels:
            try:
                shutil.copyfile(src_txt, dest_txt)
                shutil.copyfile(src_img, dest_img)
                num_kept += 1
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            except IsADirectoryError:
                print("Destination is a directory.")
            except PermissionError:
                print("Permission denied.")
        else:
            num_filtered += 1

print('kept: ' + str(num_kept))
print('removed: ' + str(num_filtered))
