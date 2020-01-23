import csv
from PIL import Image
from tqdm import tqdm
import os
import shutil


def copy_image(src, dest):
    try:
        img = Image.open(src)
        img.thumbnail((320, 320))
        width, height = img.size
        img.save(dest)
        return width, height, True
    except OSError:
        print('file missing:' + row[0] + '.jpg')
        return 0, 0, False


images_to_remove = set([])

src_dir = 'C:/Users/Florian/Documents/OIDv4_ToolKit/OID/Dataset/train/Person/'
dest_dir = 'C:/Users/Florian/Desktop/dataset_3/'

dest_folder_after_remove = 'C:/Users/Florian/Desktop/dataset_3_apply_filter/'


if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

if not os.path.exists(dest_folder_after_remove):
    os.makedirs(dest_folder_after_remove)

with open('C:/Users/Florian/Desktop/train-annotations-bbox.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    counter = 0
    for row in tqdm(csv_reader):
        if counter == 0:
            counter += 1
            continue

        # if row[0] in images_to_remove:
        #     continue

        if row[2] != '/m/01g317' or row[10] == '1' or row[11] == '1':
            images_to_remove.add(row[0])
            continue

        w, h, file_exists = copy_image(
            src_dir + row[0] + '.jpg',
            dest_dir + row[0] + '.jpg')
        if file_exists:
            x_min = int(float(row[4]) * w)
            x_max = int(float(row[5]) * w)
            y_min = int(float(row[6]) * h)
            y_max = int(float(row[7]) * h)
            file = open(dest_dir + row[0] + '.gt_data.txt', "a+")
            file.write('{} {} {} {} 0 -1 nomask 0 0\n'.format(x_min, y_min, x_max, y_max))
            file.close()


print(len(images_to_remove))


num_kept = 0
num_filtered = 0

for file in tqdm(os.listdir(dest_dir)):
    if file.endswith(".txt"):
        image_name = file.split('.', 1)[0]
        if image_name not in images_to_remove:
            src_txt = dest_dir + file
            src_img = dest_dir + image_name + '.jpg'
            dest_txt = dest_folder_after_remove + file
            dest_img = dest_folder_after_remove + image_name + '.jpg'

            shutil.copyfile(src_txt, dest_txt)
            shutil.copyfile(src_img, dest_img)
            num_kept += 1
        else:
            num_filtered += 1

print(num_kept)
print(num_filtered)
