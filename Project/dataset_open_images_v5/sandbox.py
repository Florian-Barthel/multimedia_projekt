import os
import PIL
from PIL import Image
from tqdm import tqdm


def copy_flipped_image(src, dest):
    img = Image.open(src)
    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img.save(dest)
    return


copy_flipped_image('1.jpg', '2.jpg')