# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 15:39:14 2016

@author: eggertch
"""

import os
import collections as coll

fext_img = '.jpg'


class GtItem:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    clsids = 0
    maskstr = ''
    difficult = False
    truncated = False

    def __init__(self, gt_str=''):
        if gt_str != '':
            self.from_string(gt_str)
        else:
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0
            self.clsids = -1
            self.maskstr = '_'
            self.difficult = False
            self.truncated = True

    def __hash__(self):
        return self.x1 + self.y1 * 2 ** 6 + self.x2 * 2 ** 12 + self.y2 * 2 ** 18 + self.clsids * 2 ** 24

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__dict__ == other.__dict__

    def from_string(self, gt_str):
        gt_split = gt_str.strip().split()
        self.x1 = int(gt_split[0])
        self.y1 = int(gt_split[1])
        self.x2 = int(gt_split[2])
        self.y2 = int(gt_split[3])
        self.clsids = int(gt_split[4])
        self.maskstr = gt_split[6]
        self.difficult = bool(int(gt_split[7]))
        self.truncated = bool(int(gt_split[8]))

    def __str__(self):
        gt_str = str(int(round(self.x1))) + ' '
        gt_str += str(int(round(self.y1))) + ' '
        gt_str += str(int(round(self.x2))) + ' '
        gt_str += str(int(round(self.y2))) + ' '
        gt_str += str(self.clsids) + ' '
        gt_str += '-1 '
        gt_str += self.maskstr + ' '
        gt_str += str(int(self.difficult)) + ' '
        gt_str += str(int(self.truncated))
        return gt_str

    def area(self):
        tx1 = min([self.x1, self.x2])
        tx2 = max([self.x1, self.x2])
        ty1 = min([self.y1, self.y2])
        ty2 = max([self.y1, self.y2])
        return (tx2 - tx1) * (ty2 - ty1)

    def height(self):
        ty1 = min([self.y1, self.y2])
        ty2 = max([self.y1, self.y2])
        return ty2 - ty1

    def width(self):
        tx1 = min([self.x1, self.x2])
        tx2 = max([self.x1, self.x2])
        return tx2 - tx1


class DetectionResult:
    imgname = ''
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    clsids = 0
    score = 0.0

    def __init__(self, det_str=''):
        if det_str != '':
            self.from_string(det_str)
        else:
            self.imgname = ''
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0
            self.clsids = -1
            self.score = 0.0

    def __hash__(self):
        return self.x1 + self.y1 * 2 ** 6 + self.x2 * 2 ** 12 + self.y2 * 2 ** 18 + self.clsids * 2 ** 24

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__dict__ == other.__dict__

    def from_string(self, det_str):
        det_split = det_str.strip().split()
        self.imgname = det_split[0].lower()
        self.clsids = int(det_split[1])
        self.x1 = int(float(det_split[2]))
        self.y1 = int(float(det_split[3]))
        self.x2 = int(float(det_split[4]))
        self.y2 = int(float(det_split[5]))
        self.score = float(det_split[6])

    def __str__(self):
        det_str = self.imgname + ' '
        det_str += str(self.clsids) + ' '
        det_str += str(int(round(self.x1))) + ' '
        det_str += str(int(round(self.y1))) + ' '
        det_str += str(int(round(self.x2))) + ' '
        det_str += str(int(round(self.y2))) + ' '
        det_str += str(self.score)
        return det_str

    def area(self):
        tx1 = min([self.x1, self.x2])
        tx2 = max([self.x1, self.x2])
        ty1 = min([self.y1, self.y2])
        ty2 = max([self.y1, self.y2])
        return (tx2 - tx1) * (ty2 - ty1)

    def height(self):
        ty1 = min([self.y1, self.y2])
        ty2 = max([self.y1, self.y2])
        return ty2 - ty1

    def width(self):
        tx1 = min([self.x1, self.x2])
        tx2 = max([self.x1, self.x2])
        return tx2 - tx1


class Box:
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0

    def __init__(self, box_str=''):
        if box_str != '':
            self.from_string(box_str)
        else:
            self.x1 = 0
            self.y1 = 0
            self.x2 = 0
            self.y2 = 0

    def __hash__(self):
        return self.x1 + self.y1 * 2 ** 6 + self.x2 * 2 ** 12 + self.y2 * 2 ** 18 + self.clsids * 2 ** 24

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__dict__ == other.__dict__

    def from_string(self, box_str):
        box_split = box_str.strip().split()
        self.x1 = int(box_split[0])
        self.y1 = int(box_split[1])
        self.x2 = int(box_split[2])
        self.y2 = int(box_split[3])

    def __str__(self):
        box_str = str(int(round(self.x1))) + ' '
        box_str += str(int(round(self.y1))) + ' '
        box_str += str(int(round(self.x2))) + ' '
        box_str += str(int(round(self.y2)))
        return box_str

    def area(self):
        tx1 = min([self.x1, self.x2])
        tx2 = max([self.x1, self.x2])
        ty1 = min([self.y1, self.y2])
        ty2 = max([self.y1, self.y2])
        return (tx2 - tx1) * (ty2 - ty1)

    def height(self):
        ty1 = min([self.y1, self.y2])
        ty2 = max([self.y1, self.y2])
        return ty2 - ty1

    def width(self):
        tx1 = min([self.x1, self.x2])
        tx2 = max([self.x1, self.x2])
        return tx2 - tx1


def overlap(box1, box2):
    if box1.x2 < box2.x1 or box2.x2 < box1.x1 or box1.y2 < box2.y1 or box2.y2 < box1.y1:
        return 0.0
    x_sorted = sorted([box1.x1, box1.x2, box2.x1, box2.x2])
    y_sorted = sorted([box1.y1, box1.y2, box2.y1, box2.y2])
    area_intersect = (x_sorted[2] - x_sorted[1]) * (y_sorted[2] - y_sorted[1])
    area1 = box1.area()
    area2 = box2.area()
    return float(area_intersect) / (area1 + area2 - area_intersect)


def read_gt(filename):
    gt_items = []
    with open(filename, 'r') as f:
        for fline in f:
            fline = fline.strip()
            if len(fline) > 0:
                gt_item = GtItem(fline)
                gt_items += [gt_item]
    return gt_items


def write_gt(filename, gt_items):
    with open(filename, 'w') as f:
        for gt_item in gt_items:
            print >> f, str(gt_item)


def read_prop(filename):
    prop_items = []
    with open(filename, 'r') as f:
        for fline in f:
            fline = fline.strip()
            if len(fline) > 0:
                prop_item = Box(fline)
                prop_items += [prop_item]
    return prop_items


def write_prop(filename, prop_items):
    with open(filename, 'w') as f:
        for prop_item in prop_items:
            print >> f, str(prop_item)


def read_filelist(filename):
    filelist = []
    with open(filename, 'r') as f:
        for fline in f:
            fline = fline.strip()
            if len(fline) > 0:
                filelist += [fline]
    return filelist


def get_gt_files(flickr_basedir, name_set, abs_path=True):
    filelistfile = ''
    subdir = ''
    if name_set.lower() == 'train':
        subdir = 'train'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist.txt')
    elif name_set.lower() == 'train-logosonly':
        subdir = 'train'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist-logosonly.txt')
    elif name_set.lower() == 'test':
        subdir = 'test'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist.txt')
    elif name_set.lower() == 'test-logosonly':
        subdir = 'test'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist-logosonly.txt')
    else:
        raise 'Invalid set name'
    reslist = read_filelist(filelistfile)
    reslist = map(lambda x: x[:-4] + '.gt_data.txt', reslist)
    reslist = map(lambda x: os.path.join(subdir, x), reslist)
    if abs_path:
        reslist = map(lambda x: os.path.join(flickr_basedir, x), reslist)
    return reslist


def get_img_files(flickr_basedir, name_set, abs_path=True):
    filelistfile = ''
    subdir = ''
    if name_set.lower() == 'train':
        subdir = 'train'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist.txt')
    elif name_set.lower() == 'train-logosonly':
        subdir = 'train'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist-logosonly.txt')
    elif name_set.lower() == 'test':
        subdir = 'test'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist.txt')
    elif name_set.lower() == 'test-logosonly':
        subdir = 'test'
        filelistfile = os.path.join(flickr_basedir, subdir, 'filelist-logosonly.txt')
    else:
        raise 'Invalid set name'
    reslist = read_filelist(filelistfile)
    reslist = map(lambda x: os.path.join(subdir, x), reslist)
    if abs_path:
        reslist = map(lambda x: os.path.join(flickr_basedir, x), reslist)
    return reslist


def get_dict_classid2name(flickr_basedir):
    classid2name = dict()
    classfile = os.path.join(flickr_basedir, 'className2ClassID.txt')
    with open(classfile, 'r') as f:
        for fline in f:
            fline = fline.strip()
            if len(fline) == 0:
                continue
            fdata = fline.split()
            classid2name[int(fdata[1])] = fdata[0]
    return classid2name


def get_dict_classname2id(flickr_basedir):
    classid2name = dict()
    classfile = os.path.join(flickr_basedir, 'className2ClassID.txt')
    with open(classfile, 'r') as f:
        for fline in f:
            fline = fline.strip()
            if len(fline) == 0:
                continue
            fdata = fline.split()
            classid2name[fdata[0]] = int(fdata[1])
    return classid2name


def load_detections(det_filename):
    db_det = coll.defaultdict(list)
    with open(det_filename, 'r') as f:
        for fline in f:
            fline = fline.strip()
            if len(fline) == 0:
                continue
            det_res = DetectionResult(fline)
            db_det[det_res.imgname] += [det_res]

    # for each image, sort detections by highest score
    for list_det in db_det.values():
        list_det.sort(key=lambda x: x.score, reverse=True)

    return db_det


def load_gts(flickr_basedir, subset):
    db_gt = coll.defaultdict(list)
    for root, dirs, files in os.walk(os.path.join(flickr_basedir, subset)):
        files = filter(lambda x: x.endswith('.gt_data.txt'), files)
        for fname in files:
            imgname = fname[:-len('.gt_data.txt')] + fext_img
            gts = read_gt(os.path.join(root, fname))
            db_gt[imgname.lower()] += gts
    return db_gt
