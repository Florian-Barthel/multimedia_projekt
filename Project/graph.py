import tensorflow as tf
import numpy as np
from dataSet import image_height, image_width

def probabilities_output(features, anchor_grid):
    with tf.variable_scope('probabilities'):
      ag_shape = np.shape(anchor_grid)

      features_convoluted = tf.layers.conv2d(inputs=features,
                                             filters=ag_shape[2] * ag_shape[3] * 2,
                                             kernel_size=1,
                                             strides=1,
                                             padding='same',
                                             activation=None,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                             kernel_initializer=tf.truncated_normal_initializer(),
                                             bias_initializer=tf.constant_initializer(0.0))

      probabilities = tf.reshape(features_convoluted, [tf.shape(features_convoluted)[0],
                                         ag_shape[0],
                                         ag_shape[1],
                                         ag_shape[2],
                                         ag_shape[3],
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
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='probabilities'))
    total_loss = tf.add(objective_loss, regularization_loss)

    num_labels = tf.reduce_sum(cast_labels[0])
    num_random = tf.reduce_sum(negative_examples[0])
    num_weights = tf.reduce_sum(weights[0])
    num_predicted = tf.reduce_sum(tf.argmax(cast_input[0], axis=-1))
    return total_loss, num_labels, num_weights, num_predicted






def adjustments_output(features, ag, ag_tensor):
    with tf.variable_scope('adjustments'):
        ag_shape = np.shape(ag)
        num_batch_size = tf.shape(features)[0]

        

        features_convoluted_raw = tf.layers.conv2d(inputs=tf.cast(features, tf.float64),
                                               filters=ag_shape[2] * ag_shape[3] * 4,
                                               kernel_size=3,
                                               strides=1,
                                               padding='same',
                                               activation=None,
                                               kernel_initializer=tf.truncated_normal_initializer(0.0, 1.0),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))


        #features_convoluted = tf.where(tf.math.is_nan(features_convoluted), tf.zeros_like(features_convoluted), features_convoluted)
        

        features_convoluted_raw = tf.reshape(features_convoluted_raw, [num_batch_size,
                                       ag_shape[0],
                                       ag_shape[1],
                                       ag_shape[2],
                                       ag_shape[3],
                                       4])


        # Works ok
        # clip_value = 10000.0
        # features_convoluted = tf.clip_by_value(features_convoluted, -clip_value, clip_value)
        # features_convoluted = ((features_convoluted + clip_value) / clip_value * 2.0) * 3.5
        
        features_convoluted = (tf.math.sign(features_convoluted_raw) * (tf.math.log(1.0 + tf.abs(features_convoluted_raw))))
        

        #features_convoluted = tf.nn.softmax(features_convoluted)
        #features_convoluted = (features_convoluted - tf.reduce_mean(features_convoluted)) / (tf.abs(tf.math.reduce_std(features_convoluted)) + 0.3)
        #features_convoluted = tf.math.divide_no_nan(features_convoluted - tf.reduce_min(features_convoluted), tf.abs(tf.reduce_max(features_convoluted) - tf.reduce_min(features_convoluted))  + 1.0) * 3.5
        
        #features_convoluted = tf.clip_by_value(features_convoluted, -0.5, 0.5)

        offsets = features_convoluted[..., 0:2] 
        scales = tf.abs(features_convoluted[..., 2:4])

        ag_with_batches = tf.cast(tf.tile(tf.expand_dims(ag_tensor, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float64)

        lower = ag_with_batches[..., 0:2]
        sizes = ag_with_batches[..., 2:4] - lower

        lower_offseted = lower + (lower * offsets)
        sizes_rescaled = sizes * scales

        #upper_offseted = lower_offseted + sizes_rescaled

        adjusted_bb = tf.concat([lower_offseted, sizes_rescaled], -1)

        # adjusted_bb = tf.Print(adjusted_bb, [tf.math.reduce_sum(tf.cast(tf.math.is_nan(features_convoluted), tf.int32))], summarize=12)
        # adjusted_bb = tf.Print(adjusted_bb, [tf.math.reduce_std(features_convoluted)], summarize=12)
        adjusted_bb = tf.Print(adjusted_bb, [adjusted_bb], summarize=12)
        adjusted_bb = tf.Print(adjusted_bb, ["----------"], summarize=12)

        return adjusted_bb






def adjustments_loss(adjustments, gts, overlap_labels_tensor, anchor_grid_tensor):



    num_anchors = tf.reduce_prod(tf.shape(adjustments)[1:5])
    num_images = tf.shape(adjustments)[0]

    gts = tf.cast(gts, tf.float64) * tf.constant([image_height, image_width, image_height, image_width], tf.float64)

    adjustments = tf.reshape(adjustments, [num_images, num_anchors, 4])

    mask = tf.cast(tf.reshape(overlap_labels_tensor, [num_images, num_anchors]), tf.bool)

    #adjustments = tf.tile(tf.expand_dims(adjustments, -2), [1, 1, tf.shape(gts)[1], 1])
    
    gts = gts[:,0,:]
    gts = tf.tile(tf.expand_dims(gts, 1), [1, num_anchors, 1])


    adjustments_sizes = adjustments[..., 2:4]
    gt_sizes = gts[..., 2:4] - gts[..., 0:2]

    # adjustments_sizes = tf.clip_by_value(adjustments_sizes, 5.0, 150.0)
    # gt_sizes = tf.clip_by_value(gt_sizes, 5.0, 150.0)

    offset_targets = (gts[..., 0:2] - adjustments[..., 0:2]) / adjustments_sizes
    scale_targets = tf.math.log((gt_sizes / adjustments_sizes))


    #scale_targets = tf.abs(adjustments_sizes - gt_sizes)
    
    
    #scale_targets = tf.reduce_min(scale_targets, -2)
    #scale_targets = tf.Print(scale_targets, [scale_targets], summarize=12)
    
    #return tf.nn.l2_loss(offset_targets), adjustments



    targets = tf.concat([offset_targets, scale_targets], -1)

    #targets = tf.where(tf.math.is_nan(targets), tf.zeros(tf.shape(targets), tf.float64), targets)

    #targets = tf.Print(targets, [targets], summarize=24)


    #targets = tf.reduce_min(targets, -2)
    #targets = tf.Print(targets, [tf.shape(targets)], summarize=5)

    #targets = tf.boolean_mask(targets, mask)

    
    regression_loss = tf.nn.l2_loss(targets)

    #regression_loss = tf.Print(regression_loss, [regression_loss], summarize=24)
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='adjustments'))

    loss = regression_loss + regularization_loss

    return regression_loss, adjustments











def adjustments_output_(features, anchor_grid, anchor_grid_tensor):
    with tf.variable_scope('adjustments'):
        ag_shape = np.shape(anchor_grid)

        features_convoluted = tf.layers.conv2d(inputs=features,
                                               filters=ag_shape[2] * ag_shape[3] * 4,
                                               kernel_size=10,
                                               strides=1,
                                               padding='same',
                                               activation=None,
                                               kernel_initializer=tf.constant_initializer(0.0),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))

        num_batch_size = tf.shape(features_convoluted)[0]

        adjustment_factors = tf.cast(tf.reshape(features_convoluted, [num_batch_size,
                                                       ag_shape[0],
                                                       ag_shape[1],
                                                       ag_shape[2],
                                                       ag_shape[3],
                                                       4]), tf.float64)

        # adjustment_factors = tf.where(tf.math.is_nan(adjustment_factors), tf.ones_like(adjustment_factors), adjustment_factors)

        #adjustment_factors_clipped = tf.clip_by_value(adjustment_factors, -1.0, 1.0)

        anchor_grid_with_batches = tf.cast(tf.tile(tf.expand_dims(anchor_grid_tensor, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float64)

        #bounding_boxes_overrides = tf.nn.softmax(adjustment_factors) * anchor_grid_with_batches

        offset_clip_value = np.max([image_width, image_height]) / 2.0
        #scale_clip_value = 6

        coordinates_offseted = anchor_grid_with_batches[..., 0:2] + tf.clip_by_value(adjustment_factors[..., 0:2], -offset_clip_value, offset_clip_value)
        #sizes_scaled = (anchor_grid_with_batches[..., 2:4] - anchor_grid_with_batches[..., 0:2]) * tf.clip_by_value(adjustment_factors[..., 2:4], 0.33, 1.75)
        sizes_raw = anchor_grid_with_batches[..., 2:4] - anchor_grid_with_batches[..., 0:2]
        #sizes_scaled = sizes_raw  * tf.clip_by_value(adjustment_factors[..., 2:4], 0.33, 2.0)
        #sizes_scaled = sizes_raw  + tf.clip_by_value(adjustment_factors[..., 2:4], -offset_clip_value, offset_clip_value)
        sizes_scaled = anchor_grid_with_batches[..., 2:4]  + tf.clip_by_value(adjustment_factors[..., 2:4], -offset_clip_value, offset_clip_value)
        bounding_boxes_adjusted = tf.concat([coordinates_offseted, sizes_scaled], -1)

        bounding_boxes_adjusted = tf.Print(bounding_boxes_adjusted, [adjustment_factors], summarize=12)
        # bounding_boxes_adjusted = tf.Print(bounding_boxes_adjusted, [bounding_boxes_adjusted], summarize=8)
        # bounding_boxes_adjusted = tf.Print(bounding_boxes_adjusted, [str(bounding_boxes_adjusted.shape)])
        # bounding_boxes_adjusted = tf.Print(bounding_boxes_adjusted, [str(adjustment_factors.shape)])
        # bounding_boxes_adjusted = tf.Print(bounding_boxes_adjusted, [str(anchor_grid_with_batches.shape)])

    return bounding_boxes_adjusted

def adjustments_loss_(adjustments, gt_labels_tensor, overlap_labels_tensor, anchor_grid_tensor):

    gt_labels_coordinates = tf.cast(gt_labels_tensor, tf.float64) * tf.constant([image_height, image_width, image_height, image_width], tf.float64)

    adjustments_shape = tf.shape(adjustments)
    num_anchors = tf.math.reduce_prod(adjustments_shape[1:5])
    num_batch_size = adjustments_shape[0]

    # [batch_size, num_anchors, 4]
    adjustments_flat = tf.reshape(adjustments, [num_batch_size, num_anchors, adjustments_shape[5]])



    anchor_grid_with_batches = tf.cast(tf.tile(tf.expand_dims(anchor_grid_tensor, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float64)
    anchor_grid_flat = tf.reshape(anchor_grid_with_batches, [num_batch_size, num_anchors, adjustments_shape[5]])

    num_gt_labels = tf.shape(gt_labels_coordinates)[1]
    # [batch_size, num_anchors, num_gt_labels, 4]
    adjustments_broadcasted = tf.cast(tf.tile(tf.expand_dims(adjustments_flat, -2), [1, 1, num_gt_labels, 1]), tf.float64)
    anchor_grid_broadcasted = tf.cast(tf.tile(tf.expand_dims(anchor_grid_flat, -2), [1, 1, num_gt_labels, 1]), tf.float64)
    # [batch_size, num_anchors, num_gt_labels, 4]
    gt_labels_coordinates_broadcasted = tf.tile(tf.expand_dims(gt_labels_coordinates, 1), [1, num_anchors, 1, 1])

    overlap_labels_tensor_mask = tf.cast(tf.reshape(overlap_labels_tensor, [num_batch_size, num_anchors]), tf.bool)

    

    adjustments_sizes = adjustments_broadcasted[..., 2:4]
    ag_sizes = anchor_grid_broadcasted[..., 2:4] - anchor_grid_broadcasted[..., 0:2]
    gt_sizes = gt_labels_coordinates_broadcasted[..., 2:4] - gt_labels_coordinates_broadcasted[..., 0:2]

    # adjustments_sizes = tf.clip_by_value(adjustments_broadcasted[..., 2:4] - adjustments_broadcasted[..., 0:2], 0.05, np.max([image_width, image_height]))
    # gt_sizes = tf.clip_by_value(adjustments_sizes, 0.05, np.max([image_width, image_height]))

    # offset_reg_targets = gt_labels_coordinates_broadcasted[..., 0:2] - adjustments_broadcasted[..., 0:2] / ag_sizes

    # scale_reg_targets = adjustments_sizes - gt_sizes

    # regression_targets = tf.concat([offset_reg_targets, scale_reg_targets], -1)

    regression_targets = tf.concat([gt_labels_coordinates_broadcasted[..., 0:2] - adjustments_broadcasted[..., 0:2] / ag_sizes, gt_labels_coordinates_broadcasted[..., 2:4] - adjustments_broadcasted[..., 2:4] / ag_sizes], -1)

    #regression_targets_nan_filtered = tf.where(tf.math.is_nan(regression_targets), tf.fill(tf.shape(regression_targets), float('300')), regression_targets)

    regression_targets = tf.Print(regression_targets, [regression_targets], summarize=24)

    regression_targets_l2_reduced = tf.math.reduce_sum(tf.math.abs(regression_targets), -1)

    

    regression_targets_inf_filtered = tf.math.reduce_min(regression_targets_l2_reduced, -1)

    
    regression_targets_iou_filtered = tf.boolean_mask(regression_targets_inf_filtered, overlap_labels_tensor_mask)
    
    #regression_targets_relevant_values, _ = tf.math.top_k(-regression_targets_iou_filtered)


    regression_targets_iou_filtered = tf.Print(regression_targets_iou_filtered, [tf.shape(regression_targets_iou_filtered)])
    #return regression_targets_inf_filtered

    #regression_targets_inf_filtered = tf.Print(regression_targets_inf_filtered, [regression_targets_inf_filtered])

    regression_loss = tf.math.reduce_mean(regression_targets_iou_filtered)

    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='adjustments'))

    combined_loss = tf.add(regression_loss, tf.cast(regularization_loss, tf.float64))

    #regression_loss = tf.nn.l2_loss(regression_targets_inf_filtered)

    return combined_loss, regression_targets_iou_filtered
    # 


    # regression_loss = tf.Print(regression_loss, [regression_loss])
    # regression_loss = tf.Print(regression_loss, [regression_targets])
    # regression_loss = tf.Print(regression_loss, [adjustments])

    # return regression_loss
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
