#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 18:10:03 2017

@author: eggertch
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap

import eval_script.flickr_io as fio


def remove_difficult(img2gts, img2dets):
    img2gts_filtered = dict()
    img2dets_filtered = dict()
    for imgname, gts in img2gts.items():
        dets = img2dets[imgname]
        gts_difficult = list(filter(lambda x: x.difficult, gts))
        if len(gts_difficult) > 0:
            gts_not_difficult = list(filter(lambda x: not x.difficult, gts))
            dets_filtered = []
            for det in dets:
                gts_difficult_ovl = [(x, fio.overlap(x, det)) for x in gts_difficult if x.clsids == det.clsids]
                maxovl_nondifficult = max([fio.overlap(x, det) for x in gts_not_difficult])
                if maxovl_nondifficult >= 0.5 or (
                        len(gts_difficult_ovl) > 0 and max(gts_difficult_ovl, key=lambda x: x[1])[1] < 0.5):
                    dets_filtered += [det]
            img2gts_filtered[imgname] = gts_not_difficult
            img2dets_filtered[imgname] = dets_filtered
        else:
            img2gts_filtered[imgname] = gts
            img2dets_filtered[imgname] = dets
    return img2gts_filtered, img2dets_filtered


def check_resfile_prefix(prefix):
    # makes sure that the parent directory of the file prefix exists and creates it, if necessary
    path_prefix, file_prefix = os.path.split(prefix)
    if path_prefix == '':
        path_prefix = '.'
    if not os.path.exists(path_prefix):
        print('--resfile specifies a prefix to the following directory that does not exist')
        print('  ' + path_prefix)
        os.makedirs(path_prefix)
        if os.path.exists(path_prefix):
            print('  Directory created sucessfully')
        else:
            print('  Unable to create directory')
            exit(1)


def check_imgname_mapping(img2dets, img2gts):
    # performs a sanity check to make sure that all detections can be mapped to a GT item
    for imgname in img2dets.keys():
        try:
            gts = img2gts[imgname]
        except KeyError:
            print('Could not find a GT file for following detection image name:')
            print('  ' + imgname)
            exit(1)


def clip_detections(img2dets, clip_thresh):
    if clip_thresh == None:
        return img2dets
    for imgname in img2dets.keys():
        img2dets[imgname] = list(filter(lambda x: x.score >= clip_thresh, img2dets[imgname]))


def assign_detections(dets, gts):
    det2gt = dict()
    for det in dets:
        # get all GTs that overlap at least by 0.5 and sort descending by overlap
        gts_ovl = [(gt, fio.overlap(gt, det)) for gt in gts if fio.overlap(gt, det) >= 0.5]
        if len(gts_ovl) > 0:
            gts_ovl.sort(key=lambda x: x[1], reverse=True)
            det2gt[det] = gts_ovl[0][0]  # assign detection to the GT id that overlaps most strongly
        else:
            det2gt[det] = None
    return det2gt


def pr_curve_for_class(img2dets, img2gts, clsid):
    dets2gts_flat = []
    gts_total = 0
    for imgname in img2dets.keys():
        # filter detections and gts by class
        gts_cls = list(filter(lambda x: x.clsids == clsid, img2gts[imgname]))
        dts_cls = list(filter(lambda x: x.clsids == clsid, img2dets[imgname]))
        gts_total += len(gts_cls)
        # assign detections to gts
        det2gts = assign_detections(dts_cls, gts_cls)
        # flatten all detection/gt pairs into a single list
        dets2gts_flat += [(det, gt) for det, gt in det2gts.items()]
    # globally sort by detection score (descending)
    dets2gts_flat.sort(key=lambda x: x[0].score, reverse=True)

    # tp = np.array([x[1] != None for x in dets2gts_flat], dtype=np.float32)
    tp = np.zeros(len(dets2gts_flat), dtype=np.float32)
    fp = np.zeros(len(dets2gts_flat), dtype=np.float32)
    fn = np.zeros(len(dets2gts_flat), dtype=np.float32)
    gts_seen = set([id(None)])
    for idx, det2gt in enumerate(dets2gts_flat):
        gtid = id(det2gt[1])
        gt_prev_seen = gtid in gts_seen
        # If detection was assigned to GT that was not previously assigned to anything-> TP
        tp[idx] = det2gt[1] is not None and not gt_prev_seen
        # If detection was not assigned to GT or detection was assigned to a previously assigned GT -> FP
        fp[idx] = (det2gt[1] is None) or ((det2gt[1] is not None) and gt_prev_seen)
        # all gts that have not been assigned yet -> FN
        gts_seen.add(gtid)
        fn[idx] = gts_total - (len(gts_seen) - 1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / (tp_cum + fn)
    return precision, recall


def plot_pr_curve(classname, precision, recall, ap, fileprefix, overwrite=False):
    plt.clf()
    plt.grid(axis='both')
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.0))
    plt.title('Class: {0}, Precision vs. Recall'.format(classname))
    plt.text(0.01, 0.05, 'AP = {0:>6.4f}'.format(ap), withdash=True)
    filename = '{0}_pr_{1}.png'.format(fileprefix, classname)
    if not os.path.exists(filename) or (os.path.exists(filename) and overwrite):
        plt.savefig(filename)


if __name__ == '__main__':
    #    mp.freeze_support()
    cparse = ap.ArgumentParser(
        prog='eval_detections',
        description='Performs an evaluation of detections')
    cparse.add_argument('--detection', help='Input file containing the detection results')
    cparse.add_argument('--dset_basedir', help='Base directory of the dataset')
    cparse.add_argument('--resfile', help='Path prefix to the result files to be written', default='')
    cparse.add_argument('--genplots', help='Set to true to generate PR curves for each class', type=bool, default=False)
    cparse.add_argument('--overwrite',
                        help='If set to true, result files with the specified --resfile prefix will be overwritten '
                             'without further warning',
                        type=bool, default=False)
    cparse.add_argument('--clip_thresh',
                        help='Do not evaluate PR for scores smaller than this thresholds (can be used to speed up the '
                             'evaluation at the expense of accuracy)',
                        type=float, default=None)

    # if no parameters are given, print help and exit
    if len(sys.argv) <= 1:
        cparse.print_help()
        exit(0)

    # if the user wants to generate PR-plots, he needs to specify where the files should be created
    cmdargs = cparse.parse_args()
    if cmdargs.genplots and cmdargs.resfile == '':
        print('You need to specify a valid path prefix for the --resfile parameter in order to generate plots')
        exit(1)

    # make sure the parent directory of the file prefix exists (if specified)
    if cmdargs.resfile != '':
        check_resfile_prefix(cmdargs.resfile)

    print('Loading classmap...')
    clsid2name = {0: 'person'}
    clsname2id = {'person': 0}

    print('Loading groundtruth data...')
    img2gts = fio.load_gts(cmdargs.dset_basedir, 'test')
    gts_num_images = 0
    gts_num_instances = 0
    for gts in img2gts.values():
        gts_num_images += 1
        gts_num_instances += len(gts)
    print('# of images:     {0}'.format(gts_num_images))
    print('# of detections: {0}'.format(gts_num_instances))

    print('Loading detections...')
    img2dets = fio.load_detections(cmdargs.detection)
    det_num_images = 0
    det_num_instances = 0
    for dets in img2dets.values():
        det_num_images += 1
        det_num_instances += len(dets)
    print('# of images:     {0}'.format(det_num_images))
    print('# of detections: {0}'.format(det_num_instances))

    img2gts, img2dets = remove_difficult(img2gts, img2dets)

    # Make sure that all detections have an image name that can be mapped to a GT
    check_imgname_mapping(img2dets, img2gts)

    # clip detections (if desired)
    clip_detections(img2dets, cmdargs.clip_thresh)

    # compute ap for each class
    output = ['{0:>20} | {1:<8}'.format('Classname', 'AP')]
    output += ['-' * 32]
    aps = np.zeros(len(clsid2name), dtype=np.float32)
    for clsid in clsid2name.keys():
        precision, recall = pr_curve_for_class(img2dets, img2gts, clsid)
        aps[clsid] = np.trapz(precision, recall)
        if cmdargs.genplots:
            plot_pr_curve(clsid2name[clsid], precision, recall, aps[clsid], cmdargs.resfile, cmdargs.overwrite)
        output += ['{0:>20} | {1:<6.4f}'.format(clsid2name[clsid], aps[clsid])]
    mAP = np.mean(aps)
    output += ['-' * 32]
    output += ['{0:>20} | {1:<6.4f}'.format('mAP', mAP)]

    resfile = None
    if cmdargs.resfile != '':
        respath = cmdargs.resfile + '_results.txt'
        if os.path.exists(respath) and not cmdargs.overwrite:
            print('WARNING: Output file already exists and will not overwritten (to override, set --overwrite 1)')
            print('  ' + respath)
        else:
            resfile = open(respath, 'w')
    for line_out in output:
        print(line_out)
        if resfile is not None:
            print(line_out, file=resfile)
    if resfile is not None:
        resfile.close()

    print('Done.')
