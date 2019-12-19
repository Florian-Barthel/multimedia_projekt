import nodes
import tensorflow as tf

def output(images, num_scales, num_aspect_ratios, f_rows, f_cols):
    net = nodes.mobile_net_v2()(images, training=False)

    convolution = nodes.convolution(input_tensor=net,
                                    scales=num_scales,
                                    aspect_ratios=num_aspect_ratios)

    reshape = nodes.reshape(input_tensor=convolution,
                            scales=num_scales,
                            aspect_ratios=num_aspect_ratios,
                            f_rows=f_rows,
                            f_cols=f_cols)

    offsets = output_bb_regression(net, num_scales, num_aspect_ratios, f_rows, f_cols)
    return reshape, offsets


def loss(input_tensor, labels, negative_percentage):
    return nodes.calculate_loss(input_tensor=input_tensor,
                                labels_tensor=labels,
                                negative_percentage=negative_percentage)


def output_bb_regression(images, num_scales, num_aspect_ratios, f_rows, f_cols):
    offsets = tf.layers.conv2d(inputs=images,
                               filters=num_scales * num_aspect_ratios * 4,
                               kernel_size=1,
                               strides=1,
                               padding='same',
                               activation=None,
                               kernel_initializer=tf.truncated_normal_initializer(),
                               bias_initializer=tf.constant_initializer(0.0))

    offsets = tf.reshape(offsets, [tf.shape(offsets)[0],
                                   f_rows,
                                   f_cols,
                                   num_scales,
                                   num_aspect_ratios,
                                   4])

    return offsets


def offsets_loss(calculate_offsets, gt_labels_tensor, anchor_grid):
    anchor_widths = anchor_grid[..., 3] - anchor_grid[..., 1]
    anchor_heights = anchor_grid[..., 2] - anchor_grid[..., 0]

          #y1                       #y1                    # height
    ty = (gt_labels[..., ] - anchor_grid[..., 0]) / anchor_heights 

          #x1                       #x2                    # width
    tx = (gt_labels[..., 1] - anchor_grid[..., 1]) / anchor_widths


def offsets_loss_batch(calculate_offsets, gt_labels_tensor, anchor_grid):

    # (None, 10, 10, 4, 3, 4) calculate_offsets 
    # (None, None, 4) gt_labels_tensor
    # (None, 10, 10, 4, 3, 4) anchor_grid

    # gt_labels = tf.cast(gt_labels_tensor * 320, dtype=tf.int32)

    # anchor_widths = anchor_grid[..., 3] - anchor_grid[..., 1]
    # anchor_heights = anchor_grid[..., 2] - anchor_grid[..., 0]



    #       #y1                       #y1                    # height
    # ty = (gt_labels[..., ] - anchor_grid[..., 0]) / anchor_heights 

    #       #x1                       #x2                    # width
    # tx = (gt_labels[..., 1] - anchor_grid[..., 1]) / anchor_widths


    # sy = tf.math.log(gt_labels[..., 2] - gt_labels[..., 0] / anchor_heights)

    # sx = tf.math.log(gt_labels[..., 3] - gt_labels[..., 1] / anchor_widths)



    # gt_labels_tensor.map(batch => {
    #   batch.map(gt => {
    #     gt_h = gt[2] - gt[0]
    #     gt_w = gt[3] - gt[1]

    #     sy 
    #   })
    # })




    # sx = tf.math.log(gt_labels_tensor[..., 2] - gt_labels_tensor[..., 0] / )

    # calculate_offsets = tf.Print(calculate_offsets, [str(tx.shape)])

    return calculate_offsets


    # tx = 

    # anchor_grid <> calculate_offsets => gt_labels_tensor


    # tf.shape(overlap_labels_tensor)[:-1]
    

    # tx = 