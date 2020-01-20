import tensorflow as tf
import config
from dataSet import image_height, image_width


def probabilities_output(features):
    with tf.variable_scope('probabilities'):
        features_convoluted = tf.layers.conv2d(inputs=features,
                                               filters=len(config.scales) * len(config.aspect_ratios) * 2,
                                               kernel_size=1,
                                               strides=1,
                                               padding='same',
                                               activation=None,
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                               kernel_initializer=tf.truncated_normal_initializer(),
                                               bias_initializer=tf.constant_initializer(0.0))

        probabilities = tf.reshape(features_convoluted, [tf.shape(features_convoluted)[0],
                                                         config.f_map_rows,
                                                         config.f_map_cols,
                                                         len(config.scales),
                                                         len(config.aspect_ratios),
                                                         2])

        return probabilities


def probabilities_loss(input_tensor, labels_tensor):
    cast_input = tf.cast(input_tensor, tf.float32)
    cast_labels = tf.cast(labels_tensor, tf.int32)

    if config.use_hard_negative_mining:
        weights = tf.losses.sparse_softmax_cross_entropy(
            labels=cast_labels,
            logits=cast_input,
            reduction=tf.losses.Reduction.NONE
        )
    else:
        # make random weights
        weights = tf.random.uniform(
            tf.shape(labels_tensor),
            dtype=tf.dtypes.float32
        )

    flat = tf.reshape(weights, [-1])
    values, indices = tf.nn.top_k(flat, k=tf.reduce_sum(cast_labels) * config.negative_example_factor)
    threshold = values[-1]
    negative_examples = tf.cast(weights > threshold, tf.dtypes.int32)
    weights = negative_examples + cast_labels

    objective_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=cast_labels,
        logits=cast_input,
        weights=weights
    )
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='probabilities'))
    total_loss = tf.add(objective_loss, regularization_loss)

    num_labels = tf.reduce_sum(cast_labels[0])
    num_weights = tf.reduce_sum(weights[0])
    num_predicted = tf.reduce_sum(tf.argmax(cast_input[0], axis=-1))
    return total_loss, num_labels, num_weights, num_predicted


def adjustments_output(features):
    with tf.variable_scope('adjustments'):
        num_batch_size = tf.shape(features)[0]

        features_convoluted = tf.layers.conv2d(inputs=tf.cast(features, tf.float32),
                                               filters=len(config.scales) * len(config.aspect_ratios) * 4,
                                               kernel_size=1,
                                               strides=1,
                                               padding='same',
                                               activation=None,
                                               kernel_initializer=tf.truncated_normal_initializer(),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))

        return tf.reshape(features_convoluted, [num_batch_size,
                                                config.f_map_rows,
                                                config.f_map_cols,
                                                len(config.scales),
                                                len(config.aspect_ratios),
                                                4])


def adjustments_loss(adjustments, gts, labels, ag):
    num_batch_size = tf.shape(adjustments)[0]
    num_anchors = tf.reduce_prod(tf.shape(adjustments)[1:5])
    num_max_gts = tf.shape(gts)[1]

    mask = tf.cast(tf.reshape(labels, [num_batch_size, num_anchors]), tf.bool)

    gts = tf.tile(tf.expand_dims(gts, 1), [1, num_anchors, 1, 1])
    gt_sizes = tf.abs(gts[..., 2:4] - gts[..., 0:2])

    ag_batched = tf.cast(tf.tile(tf.expand_dims(ag, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float32)
    ag_batched = tf.reshape(ag_batched, [num_batch_size, num_anchors, 4])
    ag_batched = tf.tile(tf.expand_dims(ag_batched, -2), [1, 1, num_max_gts, 1])
    anchor_grid_sizes = tf.abs(ag_batched[..., 2:4] - ag_batched[..., 0:2])

    adjustments = tf.reshape(adjustments, [num_batch_size, num_anchors, 4])
    adjustments = tf.boolean_mask(adjustments, mask)

    offset_targets = (gts[..., 0:2] - ag_batched[..., 0:2]) / anchor_grid_sizes
    scale_targets = tf.math.log((gt_sizes / anchor_grid_sizes))

    targets = tf.concat([offset_targets, scale_targets], -1)

    targets_xy1 = tf.where(tf.math.is_nan(targets[..., 0:2]),
                           tf.fill(tf.shape(targets[..., 0:2]), tf.constant(-10000, tf.float32)),
                           targets[..., 0:2])
    targets_xy2 = tf.where(tf.math.is_nan(targets[..., 2:4]),
                           tf.fill(tf.shape(targets[..., 2:4]), tf.constant(10000, tf.float32)),
                           targets[..., 2:4])

    targets = tf.concat([targets_xy1, targets_xy2], axis=-1)

    targets_sum = tf.reduce_sum(tf.abs(targets), axis=-1)
    targets_sum_min = tf.math.argmin(targets_sum, axis=-1)
    targets_sum_min_expand = tf.expand_dims(targets_sum_min, -1)

    batch_range = tf.range(num_batch_size)
    anchor_range = tf.range(num_anchors)
    grid = tf.meshgrid(anchor_range, batch_range)
    grid_expand_x = tf.cast(tf.expand_dims(grid[0], -1), tf.int64)
    grid_expand_y = tf.cast(tf.expand_dims(grid[1], -1), tf.int64)
    grid_concat = tf.concat([grid_expand_y, grid_expand_x], axis=-1)

    indices = tf.concat([grid_concat, targets_sum_min_expand], axis=-1)

    targets = tf.gather_nd(targets, indices)

    targets = tf.boolean_mask(targets, mask)

    regression_loss = tf.losses.mean_squared_error(targets, adjustments)
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='adjustments'))

    return tf.cast(regression_loss, tf.float32) + regularization_loss


def adjustments_loss_l2_loss(adjustments, input_gts, labels):
    num_batch_size = tf.shape(adjustments)[0]
    num_max_gts = tf.shape(input_gts)[1]
    num_anchors = tf.reduce_prod(tf.shape(adjustments)[1:5])

    adjustments = tf.reshape(adjustments, [num_batch_size, num_anchors, 4])
    adjustments = tf.tile(tf.expand_dims(adjustments, -2), [1, 1, num_max_gts, 1])

    gts = tf.tile(tf.expand_dims(input_gts, 1), [1, num_anchors, 1, 1])

    gts_xy1 = tf.where(tf.math.is_nan(gts[..., 0:2]),
                       tf.fill(tf.shape(gts[..., 0:2]), tf.constant(-10000, tf.float32)),
                       gts[..., 0:2])
    gts_xy2 = tf.where(tf.math.is_nan(gts[..., 2:4]),
                       tf.fill(tf.shape(gts[..., 2:4]), tf.constant(10000, tf.float32)),
                       gts[..., 2:4])

    gts = tf.concat([gts_xy1, gts_xy2], axis=-1)

    gt_sizes = tf.abs(gts[..., 2:4] - gts[..., 0:2])

    adjustments_size = tf.abs(adjustments[..., 2:4] - adjustments[..., 0:2])
    offset_difference = (gts[..., 0:2] - adjustments[..., 0:2]) / adjustments_size
    scale_difference = tf.math.log(gt_sizes / adjustments_size)

    adjustment = tf.concat([offset_difference, scale_difference], axis=-1)

    adjustment_sum = tf.reduce_sum(tf.abs(adjustment), axis=-1)
    adjustment_sum_min = tf.math.argmin(adjustment_sum, axis=-1)
    adjustment_sum_min_expand = tf.expand_dims(adjustment_sum_min, -1)

    batch_range = tf.range(num_batch_size)
    anchor_range = tf.range(num_anchors)
    grid = tf.meshgrid(anchor_range, batch_range)
    grid_expand_x = tf.cast(tf.expand_dims(grid[0], -1), tf.int64)
    grid_expand_y = tf.cast(tf.expand_dims(grid[1], -1), tf.int64)
    grid_concat = tf.concat([grid_expand_y, grid_expand_x], axis=-1)

    indices = tf.concat([grid_concat, adjustment_sum_min_expand], axis=-1)

    min_adjustments = tf.gather_nd(adjustment, indices)

    mask = tf.cast(tf.reshape(labels, [num_batch_size, num_anchors]), tf.bool)
    filtered_min_adjustments = tf.boolean_mask(min_adjustments, mask)

    # regression_loss = tf.nn.l2_loss(filtered_min_adjustments)
    regression_loss = tf.math.square(tf.reduce_sum(filtered_min_adjustments))
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='adjustments'))
    return regression_loss + regularization_loss, filtered_min_adjustments, gts, grid_concat, min_adjustments, offset_difference, scale_difference, adjustments_size


def adjustments_loss_working(adjustments, gts, labels, ag):
    num_batch_size = tf.shape(adjustments)[0]
    num_anchors = tf.reduce_prod(tf.shape(adjustments)[1:5])
    num_max_gts = tf.shape(gts)[1]

    mask = tf.cast(tf.reshape(labels, [num_batch_size, num_anchors]), tf.bool)

    # why norm backwards
    gts = tf.cast(gts, tf.float32)
    gts = tf.tile(tf.expand_dims(gts, 1), [1, num_anchors, 1, 1])
    gt_sizes = gts[..., 2:4] - gts[..., 0:2]

    # anchor grid auf batch_size multiplizieren
    ag_batched = tf.cast(tf.tile(tf.expand_dims(ag, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float32)
    ag_batched = tf.reshape(ag_batched, [num_batch_size, num_anchors, 4])
    ag_batched = tf.tile(tf.expand_dims(ag_batched, -2), [1, 1, num_max_gts, 1])
    # shape = (batch_size, num_anchors, num_max_gts, 4)

    anchor_grid_sizes = ag_batched[..., 2:4] - ag_batched[..., 0:2]
    # shape = (batch_size, num_anchors, num_max_gts, 2)

    adjustments = tf.reshape(adjustments, [num_batch_size, num_anchors, 4])
    adjustments = tf.tile(tf.expand_dims(adjustments, -2), [1, 1, num_max_gts, 1])
    # shape = (batch_size, num_anchors, num_max_gts, 4)

    offset_targets = (gts[..., 0:2] - ag_batched[..., 0:2]) / anchor_grid_sizes
    scale_targets = tf.math.log((gt_sizes / anchor_grid_sizes) + 0.01)

    targets = tf.concat([offset_targets, scale_targets], -1)
    # shape = (batch_size, num_anchors, num_max_gts, 4)

    targets = tf.where(tf.math.is_nan(targets), tf.fill(tf.shape(targets), tf.constant(float("Inf"), tf.float32)), targets)

    differences = tf.abs(targets - adjustments)
    differences_summed = tf.reduce_sum(differences, -1)
    # shape = (batch_size, num_anchors, num_max_gts)
    differences_closest = tf.reduce_min(differences_summed, -1)
    # shape = (batch_size, num_anchors)
    targets = tf.boolean_mask(differences_closest, mask)

    regression_loss = tf.reduce_sum(targets)

    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='adjustments'))

    return tf.cast(regression_loss, tf.float32) + regularization_loss