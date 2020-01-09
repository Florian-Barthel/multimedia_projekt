import os
from scipy.special import softmax
import numpy as np
from annotationRect import AnnotationRect
import geometry
import eval_script.flickr_io as fio
import eval_detections as eval

# Preparing data for the evaluation script
# The script can be run from the Project directory by invoking:
# python eval_script\eval_detections.py --detection eval_script/detections.txt --dset_basedir dataset_mmp

detections_path = 'eval_script/detections.txt'
default_fg = 0.01

# Non-maximum-suppression with default threshold of 0.3 (IoU)
# Input: dict of boxes AnnotationRect:Score, (optional) IoU threshold
# Output: dict of resulting AnnotationRect:Score boxes after suppression
def non_maximum_suppression(boxes, threshold=0.3):
    output = {}
    # Loop until boxes is empty
    while boxes:
        # Find box with highest score and add to output
        max_box = max(boxes, key=boxes.get)
        output[max_box] = boxes[max_box]
        boxes.pop(max_box)
        # Remove all boxes with IoU > threshold
        for b in list(boxes):
            if geometry.iou(max_box, b) > threshold:
                boxes.pop(b)
    return output


# Creating dict of boxes AnnotationRect:Score from the output and the anchor grid
def create_boxes_dict(data, anchor_grid, fg_threshold=default_fg):
    boxes_dict = {}
    scores = []
    calc_softmax = softmax(data, axis=-1)
    foreground = np.delete(calc_softmax, [0], axis=-1)
    # Get the scores from the data
    shape = foreground.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    for m in range(shape[4]):
                        if foreground[i, j, k, l, m] > fg_threshold:
                            scores.append(foreground[i, j, k, l, m])
    # Get the boxes from the data
    filtered_indices = np.where(foreground > fg_threshold)
    remove_last = filtered_indices[:4]
    max_boxes = anchor_grid[remove_last]
    boxes = [AnnotationRect(*max_boxes[i]) for i in range(max_boxes.shape[0])]
    for i in range(len(boxes)):
        boxes_dict[boxes[i]] = scores[i]
    return boxes_dict


# Save detections in text file; each line in format:
# <Image name> <class_id> <x1> <y1> <x2> <y2> <score>
def save_boxes(boxes, image_path):
    file = open(detections_path, "a+")
    for b in boxes:
        file.write("{name} 0 {x1} {y1} {x2} {y2} {score}\n".format(name=image_path.split("/")[-1],
                                                                   x1=int(b.x1),
                                                                   y1=int(b.y1),
                                                                   x2=int(b.x2),
                                                                   y2=int(b.y2),
                                                                   score=boxes[b]))
    file.close()


# Clears the detections file located at detections_path
def clear_detections():
    open(detections_path, "w+").close()


# Prepares detections from the output and anchor_grid applying non-maximum-suppression
# and saving the resulting detections to disk
def prepare_detections(output, anchor_grid, image_paths, num_test_images, nms_threshold=0.3, fg_threshold=default_fg):
    clear_detections()
    nms_boxes = []
    for i in range(num_test_images):
        boxes_dict = create_boxes_dict(output[i], anchor_grid, fg_threshold)
        nms_boxes.append(non_maximum_suppression(boxes_dict, nms_threshold))
        save_boxes(nms_boxes[i], image_paths[i])
    return nms_boxes

# Using code from the supplied eval_detections.py to evaluate the model in runtime
def evaluate(detection, dset_basedir, resfile='', genplots=False, overwrite=False, clip_thresh=None):
    if genplots and resfile == '':
        print('You need to specify a valid path prefix for the --resfile parameter in order to generate plots')
        exit(1)

    # make sure the parent directory of the file prefix exists (if specified)
    if resfile != '':
        eval.check_resfile_prefix(resfile)

    print('Loading classmap...')
    clsid2name = {0: 'person'}
    clsname2id = {'person': 0}

    print('Loading groundtruth data...')
    img2gts = fio.load_gts(dset_basedir, 'test')
    gts_num_images = 0
    gts_num_instances = 0
    for gts in img2gts.values():
        gts_num_images += 1
        gts_num_instances += len(gts)
    print('# of images:     {0}'.format(gts_num_images))
    print('# of detections: {0}'.format(gts_num_instances))

    print('Loading detections...')
    img2dets = fio.load_detections(detection)
    det_num_images = 0
    det_num_instances = 0
    for dets in img2dets.values():
        det_num_images += 1
        det_num_instances += len(dets)
    print('# of images:     {0}'.format(det_num_images))
    print('# of detections: {0}'.format(det_num_instances))

    img2gts, img2dets = eval.remove_difficult(img2gts, img2dets)

    # Make sure that all detections have an image name that can be mapped to a GT
    eval.check_imgname_mapping(img2dets, img2gts)

    # clip detections (if desired)
    eval.clip_detections(img2dets, clip_thresh)

    # compute ap for each class
    output = ['{0:>20} | {1:<8}'.format('Classname', 'AP')]
    output += ['-' * 32]
    aps = np.zeros(len(clsid2name), dtype=np.float32)
    for clsid in clsid2name.keys():
        precision, recall = eval.pr_curve_for_class(img2dets, img2gts, clsid)
        aps[clsid] = np.trapz(precision, recall)
        if genplots:
            eval.plot_pr_curve(clsid2name[clsid], precision, recall, aps[clsid], cmdargs.resfile, cmdargs.overwrite)
        output += ['{0:>20} | {1:<6.4f}'.format(clsid2name[clsid], aps[clsid])]
    mAP = np.mean(aps)
    output += ['-' * 32]
    output += ['{0:>20} | {1:<6.4f}'.format('mAP', mAP)]
    out_resfile = None
    if resfile != '':
        respath = resfile + '_results.txt'
        if os.path.exists(respath) and not overwrite:
            print('WARNING: Output file already exists and will not overwritten (to override, set --overwrite 1)')
            print('  ' + respath)
        else:
            out_resfile = open(respath, 'w')
    for line_out in output:
        print(line_out)
        if out_resfile is not None:
            print(line_out, file=out_resfile)
    if out_resfile is not None:
        out_resfile.close()
