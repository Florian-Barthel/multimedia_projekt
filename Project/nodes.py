import tensorflow as tf
import config


def mobile_net_v2():
    return tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0,
                                                          include_top=False,
                                                          weights='imagenet',
                                                          pooling=None)


def convolution(input_tensor):
    return tf.layers.conv2d(
        inputs=input_tensor,
        filters=len(config.scales) * len(config.aspect_ratios) * 2,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
        kernel_initializer=tf.truncated_normal_initializer(),
        bias_initializer=tf.constant_initializer(0.0))


# def weighted_sum(input_tensor_1, input_tensor_2):
#     a = tf.Variable(initial_value=tf.constant(0.5), trainable=True, dtype=tf.float32)
#     b = tf.Variable(initial_value=tf.constant(0.5), trainable=True, dtype=tf.float32)
#     return tf.math.add(tf.math.multiply(input_tensor_1, a), tf.math.multiply(input_tensor_2, b))


def reshape(input_tensor):
    result = tf.reshape(input_tensor, [tf.shape(input_tensor)[0],
                                       config.f_map_rows,
                                       config.f_map_cols,
                                       len(config.scales),
                                       len(config.aspect_ratios),
                                       2])
    return result


def calculate_loss(input_tensor, labels_tensor):
    cast_input = tf.cast(input_tensor, tf.float32)
    cast_labels = tf.cast(labels_tensor, tf.int32)

    # make random weights
    random_weights = tf.random.uniform(
        tf.shape(labels_tensor),
        dtype=tf.dtypes.float32
    )
    flat = tf.reshape(random_weights, [-1])
    values, indices = tf.nn.top_k(flat, k=tf.reduce_sum(cast_labels) * config.negative_example_factor)
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
    num_weights = tf.reduce_sum(weights[0])
    num_predicted = tf.reduce_sum(tf.argmax(cast_input[0], axis=-1))
    return total_loss, num_labels, num_weights, num_predicted
