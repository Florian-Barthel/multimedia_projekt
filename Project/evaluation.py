import geometry

# Input: dict of boxes AnnotationRect:Score, (optional) IoU threshold
def non_maximum_suppression(boxes, threshold=0.3):
    output = []
    # Loop until boxes is empty
    while boxes:
        # Find box with highest score and add to output
        max_box = max(boxes, key=boxes.get)
        output.append(max_box)
        boxes.pop(max_box)
        # Remove all boxes with IoU > threshold
        for b in boxes:
            if geometry.iou(max_box, b) > threshold:
                boxes.pop(b)
    return output



