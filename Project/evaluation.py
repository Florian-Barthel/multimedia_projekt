import os
from scipy.special import softmax
import numpy as np
from annotationRect import AnnotationRect
import geometry
from datetime import datetime
import config
# Preparing data for the evaluation script
# The script can be run from the Project directory by invoking:
# python eval_script\eval_detections.py --detection eval_script/detections.txt --dset_basedir dataset_mmp

current_time = datetime.now()


# Non-maximum-suppression with default threshold of 0.3 (IoU)
# Input: dict of boxes AnnotationRect:Score, (optional) IoU threshold
# Output: dict of resulting AnnotationRect:Score boxes after suppression
def non_maximum_suppression(boxes):
    output = {}
    # Loop until boxes is empty
    while boxes:
        # Find box with highest score and add to output
        max_box = max(boxes, key=boxes.get)
        output[max_box] = boxes[max_box]
        boxes.pop(max_box)
        # Remove all boxes with IoU >= threshold
        for b in list(boxes):
            if geometry.iou(max_box, b) >= config.nms_threshold:
                boxes.pop(b)
    return output


# Creating dict of boxes AnnotationRect:Score from the output and the anchor grid
def create_boxes_dict(data, anchor_grid, foreground_threshold=0.05):
    boxes_dict = {}
    calc_softmax = softmax(data, axis=-1)
    foreground = np.delete(calc_softmax, [0], axis=-1)
    indices = np.where(foreground > foreground_threshold)[:4]
    scores = foreground[indices]
    max_boxes = anchor_grid[indices]
    # boxes = [AnnotationRect(*max_boxes[i]) for i in range(max_boxes.shape[0])]
    for i in range(len(scores)):
        boxes_dict[AnnotationRect(*max_boxes[i])] = scores[i, 0]
    return boxes_dict


# Save detections in text file; each line in format:
# <Image name> <class_id> <x1> <y1> <x2> <y2> <score>
def save_boxes(boxes, image_path, detection_file):
    file = open(detection_file, "a+")
    for b in boxes:
        file.write("{name} 0 {x1} {y1} {x2} {y2} {score}\n".format(name=image_path.split("/")[-1],
                                                                   x1=int(b.x1),
                                                                   y1=int(b.y1),
                                                                   x2=int(b.x2),
                                                                   y2=int(b.y2),
                                                                   score=boxes[b]))
    file.close()


# Prepares detections from the output and anchor_grid applying non-maximum-suppression
# and saving the resulting detections to disk

def prepare_detections(output, anchor_grid, image_paths, detection_file):
    for i in range(len(image_paths)):
        # check if anchor_grid is static or from bbr
        if len(np.shape(anchor_grid)) == 6:
            ag = anchor_grid[i]
        else:
            ag = anchor_grid
        boxes_dict = create_boxes_dict(output[i], ag)
        nms = non_maximum_suppression(boxes_dict)
        save_boxes(nms, image_paths[i], detection_file)
