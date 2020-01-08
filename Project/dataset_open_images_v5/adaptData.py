import csv
from PIL import Image
from tqdm import tqdm


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


with open('train-annotations-bbox.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    counter = 0
    for row in tqdm(csv_reader):
        if counter == 0 or row[2] != '/m/01g317':
            pass
        else:

            w, h, file_exists = copy_image(
                'C:/Users/Florian/Documents/OIDv4_ToolKit/OID/Dataset/train/Person/' + row[0] + '.jpg',
                'C:/Users/Florian/Desktop/train_v4/' + row[0] + '.jpg')
            if file_exists:
                x_min = int(float(row[4]) * w)
                x_max = int(float(row[5]) * w)
                y_min = int(float(row[6]) * h)
                y_max = int(float(row[7]) * h)
                file = open('C:/Users/Florian/Desktop/train_v4/' + row[0] + '.gt_data.txt', "a+")
                file.write('{} {} {} {} 0 -1 nomask 0 0\n'.format(x_min, y_min, x_max, y_max))
                file.close()

        counter += 1
