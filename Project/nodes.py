import tensorflow as tf


def mobile_net_v2():
    return tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0,
                                                          include_top=False,
                                                          weights='imagenet',
                                                          pooling=None)


def convolution(input_tensor, scales, aspect_ratios):
    return tf.layers.conv2d(
        inputs=input_tensor,
        filters=scales * aspect_ratios * 2,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005),
        kernel_initializer=tf.truncated_normal_initializer(),
        bias_initializer=tf.constant_initializer(0.0))


def reshape(input_tensor, scales, aspect_ratios, f_cols, f_rows):
    result = tf.reshape(input_tensor, [tf.shape(input_tensor)[0],
                                       f_rows,
                                       f_cols,
                                       scales,
                                       aspect_ratios,
                                       2])
    return result


def calculate_loss(input_tensor, labels_tensor, negative_example_factor=10):
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
