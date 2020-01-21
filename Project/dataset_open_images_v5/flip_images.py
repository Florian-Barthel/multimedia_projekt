import os
import PIL
from PIL import Image
from tqdm import tqdm


def copy_flipped_image(src, dest):
    img = Image.open(src)
    w, h = img.size
    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img.save(dest)
    return w


src_folder = '../../datasets/dataset_2_crowd_min_ratio'
dest_folder = '../../datasets/dataset_2_crowd_min_ratio_flipped'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)


for file in tqdm(os.listdir(src_folder)):
    if file.endswith(".txt"):
        src_txt = src_folder + '/' + file
        src_img = src_folder + '/' + file.split('.', 1)[0] + '.jpg'
        dest_txt = dest_folder + '/' + file.split('.', 1)[0] + '_flip.gt_data.txt'
        dest_img = dest_folder + '/' + file.split('.', 1)[0] + '_flip.jpg'
        width = copy_flipped_image(src_img, dest_img)
        file_read = open(src_txt, 'r')
        if file_read.mode == 'r':
            lines = file_read.readlines()
            for line in lines:
                tokens = line.split(' ')
                x1 = width - int(tokens[0])
                y1 = int(tokens[1])
                x2 = width - int(tokens[2])
                y2 = int(tokens[3])
                file_write = open(dest_txt, "a+")
                file_write.write('{} {} {} {} 0 -1 nomask 0 0\n'.format(x1, y1, x2, y2))
                file_write.close()
