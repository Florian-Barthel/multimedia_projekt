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

        features_convoluted = tf.layers.conv2d(inputs=tf.cast(features, tf.float64),
                                               filters=ag_shape[2] * ag_shape[3] * 4,
                                               kernel_size=1,
                                               strides=1,
                                               padding='same',
                                               activation=None,
                                               kernel_initializer=tf.constant_initializer(0.0),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
                                               bias_initializer=tf.constant_initializer(0.0),
                                               bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005)) + 1.0
        
        return tf.reshape(features_convoluted, [num_batch_size,
                                                ag_shape[0],
                                                ag_shape[1],
                                                ag_shape[2],
                                                ag_shape[3],
                                                4])


def adjustments_loss(adjustments, gts, labels, ag):
    num_batch_size = tf.shape(adjustments)[0]
    num_anchors = tf.reduce_prod(tf.shape(adjustments)[1:5])
    num_max_gts = tf.shape(gts)[1]

    mask = tf.cast(tf.reshape(labels, [num_batch_size, num_anchors]), tf.bool)

    gts = tf.cast(gts, tf.float64) * tf.constant([image_height, image_width, image_height, image_width], tf.float64)
    gts = tf.tile(tf.expand_dims(gts, 1), [1, num_anchors, 1, 1])
    gt_sizes = gts[..., 2:4] - gts[..., 0:2]

    ag_batched = tf.cast(tf.tile(tf.expand_dims(ag, 0), [num_batch_size, 1, 1, 1, 1, 1]), tf.float64)
    ag_batched = tf.reshape(ag_batched, [num_batch_size, num_anchors, 4])
    ag_batched = tf.tile(tf.expand_dims(ag_batched, -2), [1, 1, num_max_gts, 1])
    anchor_grid_sizes = ag_batched[..., 2:4] - ag_batched[..., 0:2]
    
    adjustments = tf.reshape(adjustments, [num_batch_size, num_anchors, 4])
    adjustments = tf.boolean_mask(adjustments, mask)

    offset_targets = (gts[..., 0:2] - ag_batched[..., 0:2]) / anchor_grid_sizes
    scale_targets = tf.math.log((gt_sizes / anchor_grid_sizes))

    targets = tf.concat([offset_targets, scale_targets], -1)
    targets = tf.where(tf.math.is_nan(targets), tf.fill(tf.shape(targets), tf.constant(float("Inf"), tf.float64)), targets)
    targets = tf.reduce_min(targets, -2)
    targets = tf.boolean_mask(targets, mask)

    regression_loss = tf.losses.mean_squared_error(targets, adjustments)
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='adjustments'))

    return tf.cast(regression_loss, tf.float64) + regularization_loss