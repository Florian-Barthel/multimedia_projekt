import numpy as np
from annotationRect import AnnotationRect


def area_intersection(rect1, rect2):
    overlap_width = min(rect1.x2, rect2.x2) - max(rect1.x1, rect2.x1)
    overlap_height = min(rect1.y2, rect2.y2) - max(rect1.y1, rect2.y1)
    return max(0, overlap_width) * max(0, overlap_height)


def area_union(rect1, rect2):
    return rect1.area() + rect2.area() - area_intersection(rect1, rect2)


def iou(rect1, rect2):
    return area_intersection(rect1, rect2) / area_union(rect1, rect2)


def anchor_max_gt_overlaps(anchor_grid, gts):
    output = np.zeros(anchor_grid.shape[:-1])

    for y in range(anchor_grid.shape[0]):
        for x in range(anchor_grid.shape[1]):
            for s in range(anchor_grid.shape[2]):
                for a in range(anchor_grid.shape[3]):
                    anchor_box = AnnotationRect(anchor_grid[x, y, s, a, 0],
                                                anchor_grid[x, y, s, a, 1],
                                                anchor_grid[x, y, s, a, 2],
                                                anchor_grid[x, y, s, a, 3])
                    output[x][y][s][a] = max([iou(anchor_box, gt) for gt in gts])
    return output
