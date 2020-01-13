import numpy as np


def anchor_grid(f_map_rows, f_map_cols, scale_factor, scales, aspect_ratios):
    output = np.ones((f_map_rows,
                      f_map_cols,
                      len(scales),
                      len(aspect_ratios),
                      4)
                     )

    for y in range(f_map_rows):
        for x in range(f_map_cols):
            for s in range(len(scales)):
                for r in range(len(aspect_ratios)):
                    scale = scales[s]
                    aspect_ratio = aspect_ratios[r]
                    box_radius_x = int(scale / 2)
                    box_radius_y = int((scale * aspect_ratio) / 2)
                    center_x = int(x * scale_factor + scale_factor / 2)
                    center_y = int(y * scale_factor + scale_factor / 2)

                    x1 = center_x - box_radius_x
                    y1 = center_y - box_radius_y
                    x2 = center_x + box_radius_x
                    y2 = center_y + box_radius_y

                    output[y][x][s][r] = (x1, y1, x2, y2)
    return output
