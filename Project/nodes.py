import tensorflow as tf

mobile_net_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0,
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


def normalize(input_tensor):
    cast = tf.cast(input_tensor, tf.float32)
    norm = cast / 255
    return norm


def reshape(input_tensor, batch_size, scales, aspect_ratios):
    return tf.reshape(input_tensor, [batch_size,
                                     tf.shape(input_tensor)[1],
                                     tf.shape(input_tensor)[2],
                                     scales,
                                     aspect_ratios,
                                     2])


def calculate_loss(input_tensor, labels_tensor):
    cast_output = tf.cast(input_tensor, tf.float32)
    cast_labels = tf.cast(labels_tensor, tf.int32)
    objective_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=cast_labels,
        logits=cast_output
    )
    # regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total_loss = tf.add(objective_loss, regularization_loss)
    return objective_loss
