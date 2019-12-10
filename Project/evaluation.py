from scipy.special import softmax
import numpy as np
from annotationRect import AnnotationRect
import geometry

# Preparing data for the evaluation script
# The script can be run from the Project directory by invoking:
# python eval_script\eval_detections.py --detection eval_script/detections.txt --dset_basedir dataset_mmp

detections_path = 'eval_script/detections.txt'


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
def create_boxes_dict(data, anchor_grid, fg_threshold=0.01):
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
def prepare_detections(output, anchor_grid, image_paths, num_test_images, nms_threshold=0.3):
    clear_detections()
    for i in range(num_test_images):
        boxes_dict = create_boxes_dict(output[i], anchor_grid)
        nms = non_maximum_suppression(boxes_dict, nms_threshold)
        save_boxes(nms, image_paths[i])
