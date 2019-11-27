import tensorflow as tf

mobile_net_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0,
                                                               input_shape=(320, 320, 3),
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
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        bias_initializer=tf.constant_initializer(0.0))


def reshape(input_tensor, scales, aspect_ratios, f_cols, f_rows):
    result = tf.reshape(input_tensor, [tf.shape(input_tensor)[0],
                                       f_rows,
                                       f_cols,
                                       scales,
                                       aspect_ratios,
                                       2])
    return result


def calculate_loss(input_tensor, labels_tensor, negative_percentage):
    cast_input = tf.cast(input_tensor, tf.float32)
    cast_labels = tf.cast(labels_tensor, tf.int32)
    random_labels = tf.random.uniform(
        (tf.shape(labels_tensor)),
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32
    )
    reduced_labels = tf.where(condition=random_labels < negative_percentage, x=random_labels * 0, y=random_labels)
    labels_filtered = tf.math.logical_or(x=tf.cast(reduced_labels, tf.bool), y=tf.cast(cast_labels, tf.bool))
    cast_labels_filtered = tf.cast(labels_filtered, tf.int32)

    objective_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=cast_labels,
        logits=cast_input,
        weights=cast_labels_filtered
    )
    # regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total_loss = tf.add(objective_loss, regularization_loss)
    return objective_loss
