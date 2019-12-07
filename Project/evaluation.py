from scipy.special import softmax
import numpy as np
from annotationRect import AnnotationRect

import geometry

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
def create_boxes_dict(data, anchor_grid, fg_threshold=0.7):
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
