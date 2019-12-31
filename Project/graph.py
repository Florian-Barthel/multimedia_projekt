import mobilenet
import tensorflow as tf
from dataSet import image_height, image_width

def probabilities_output(images, num_scales, num_aspect_ratios, f_rows, f_cols):
    features = mobilenet.mobile_net_v2()(images, training=False)

    features_convoluted = tf.layers.conv2d(inputs=features,
                                           filters=num_scales * num_aspect_ratios * 2,
                                           kernel_size=1,
                                           strides=1,
                                           padding='same',
                                           activation=None,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                           kernel_initializer=tf.truncated_normal_initializer(),
                                           bias_initializer=tf.constant_initializer(0.0))

    probabilities = tf.reshape(features_convoluted, [tf.shape(features_convoluted)[0],
                                       f_rows,
                                       f_cols,
                                       num_scales,
                                       num_aspect_ratios,
                                       2])

    return probabilities


def probabilities_loss(input_tensor, labels_tensor, negative_example_factor=10):
    cast_input = tf.cast(input_tensor, tf.float32)
    cast_labels = tf.cast(labels_tensor, tf.int32)

    # make random weights
    random_weights = tf.random.uniform(
        tf.shape(labels_tensor),
        dtype=tf.dtypes.float32
    )

    flat = tf.reshape(random_weights, [-1])
    values, indices = tf.nn.top_k(flat, k=tf.reduce_sum(cast_labels) * negative_example_factor)
    threshold = values[-1]
    negative_examples = tf.cast(random_weights > threshold, tf.dtypes.int32)
    weights = negative_examples + cast_labels

    objective_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=cast_labels,
        logits=cast_input,
        weights=weights
    )
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = tf.add(objective_loss, regularization_loss)

    num_labels = tf.reduce_sum(cast_labels[0])
    num_random = tf.reduce_sum(negative_examples[0])
    num_weights = tf.reduce_sum(weights[0])
    num_predicted = tf.reduce_sum(tf.argmax(cast_input[0], axis=-1))
    return total_loss, num_labels, num_weights, num_predicted





def adjustments_output(images, num_scales, num_aspect_ratios, f_rows, f_cols):
    features = mobilenet.mobile_net_v2()(images, training=False)

    features_convoluted = tf.layers.conv2d(inputs=features,
                                           filters=num_scales * num_aspect_ratios * 4,
                                           kernel_size=1,
                                           strides=1,
                                           padding='same',
                                           activation=None,
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                           kernel_initializer=tf.truncated_normal_initializer(),
                                           bias_initializer=tf.constant_initializer(0.0))

    adjustments = tf.reshape(features_convoluted, [tf.shape(features_convoluted)[0],
                                                   f_rows,
                                                   f_cols,
                                                   num_scales,
                                                   num_aspect_ratios,
                                                   4])

    return adjustments

def adjustments_loss(input_tensor, gt_labels_tensor, anchor_grid_tensor):
    gt_labels_coordinates = gt_labels_tensor * tf.constant([image_height, image_width, image_height, image_width], tf.float32)

    adjustments = input_tensor * tf.cast(anchor_grid_tensor, tf.float32)

    adjustments_shape = tf.shape(adjustments)
    num_anchors = tf.math.reduce_prod(adjustments_shape[1:5])

    # [batch_size, num_anchors, 4]
    adjustments_flat = tf.reshape(adjustments, [adjustments_shape[0], num_anchors, adjustments_shape[5]])

    num_gt_labels = tf.shape(gt_labels_coordinates)[1]
    # [batch_size, num_anchors, num_gt_labels, 4]
    adjustments_broadcasted = tf.tile(tf.expand_dims(adjustments_flat, -2), [1, 1, num_gt_labels, 1])
    # [batch_size, num_anchors, num_gt_labels, 4]
    gt_labels_coordinates_broadcasted = tf.tile(tf.expand_dims(gt_labels_coordinates, 1), [1, num_anchors, 1, 1])

    
    adjustments_size = adjustments_broadcasted[..., 2:4] - adjustments_broadcasted[..., 0:2]
    gt_size = gt_labels_coordinates_broadcasted[..., 2:4] - gt_labels_coordinates_broadcasted[..., 0:2]

    offset_reg_targets = (gt_labels_coordinates_broadcasted[..., 0:2] - adjustments_broadcasted[..., 0:2]) / adjustments_size
    scale_reg_targets = tf.math.log(gt_size / adjustments_size)


    combined_regression_targets = tf.concat([offset_reg_targets, scale_reg_targets], -1)

    combined_regression_targets = tf.where(tf.math.is_nan(combined_regression_targets), tf.zeros_like(combined_regression_targets), combined_regression_targets)

    return tf.nn.l2_loss(combined_regression_targets)
    #return tf.reshape(tf.tile(, [1,1, num_gt_labes]), [10, 1600, 4, num_gt_labes])
    





# def output_bb_regression(images, num_scales, num_aspect_ratios, f_rows, f_cols):
#     offsets = tf.layers.conv2d(inputs=images,
#                                filters=num_scales * num_aspect_ratios * 4,
#                                kernel_size=1,
#                                strides=1,
#                                padding='same',
#                                activation=None,
#                                kernel_initializer=tf.truncated_normal_initializer(),
#                                bias_initializer=tf.constant_initializer(0.0))

#     offsets = tf.reshape(offsets, [tf.shape(offsets)[0],
#                                    f_rows,
#                                    f_cols,
#                                    num_scales,
#                                    num_aspect_ratios,
#                                    4])

#     return offsets


# def offsets_loss(calculate_offsets, gt_labels_tensor, anchor_grid):
#     # anchor_widths = anchor_grid[..., 3] - anchor_grid[..., 1]
#     # anchor_heights = anchor_grid[..., 2] - anchor_grid[..., 0]

#     #       #y1                       #y1                    # height
#     # ty = (gt_labels[..., ] - anchor_grid[..., 0]) / anchor_heights 

#     #       #x1                       #x2                    # width
#     # tx = (gt_labels[..., 1] - anchor_grid[..., 1]) / anchor_widths

#     return calculate_offsets


# def offsets_loss_batch(calculate_offsets, gt_labels_tensor, anchor_grid):

#     # (None, 10, 10, 4, 3, 4) calculate_offsets 
#     # (None, None, 4) gt_labels_tensor
#     # (None, 10, 10, 4, 3, 4) anchor_grid

#     # gt_labels = tf.cast(gt_labels_tensor * 320, dtype=tf.int32)

#     # anchor_widths = anchor_grid[..., 3] - anchor_grid[..., 1]
#     # anchor_heights = anchor_grid[..., 2] - anchor_grid[..., 0]



#     #       #y1                       #y1                    # height
#     # ty = (gt_labels[..., ] - anchor_grid[..., 0]) / anchor_heights 

#     #       #x1                       #x2                    # width
#     # tx = (gt_labels[..., 1] - anchor_grid[..., 1]) / anchor_widths


#     # sy = tf.math.log(gt_labels[..., 2] - gt_labels[..., 0] / anchor_heights)

#     # sx = tf.math.log(gt_labels[..., 3] - gt_labels[..., 1] / anchor_widths)



#     # gt_labels_tensor.map(batch => {
#     #   batch.map(gt => {
#     #     gt_h = gt[2] - gt[0]
#     #     gt_w = gt[3] - gt[1]

#     #     sy 
#     #   })
#     # })




#     # sx = tf.math.log(gt_labels_tensor[..., 2] - gt_labels_tensor[..., 0] / )

#     # calculate_offsets = tf.Print(calculate_offsets, [str(tx.shape)])

#     return calculate_offsets


#     # tx = 

#     # anchor_grid <> calculate_offsets => gt_labels_tensor


#     # tf.shape(overlap_labels_tensor)[:-1]
    

#     # tx = 
