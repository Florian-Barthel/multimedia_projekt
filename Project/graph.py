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
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        bias_initializer=tf.constant_initializer(0.0))


def normalize(input_tensor):
    cast = tf.cast(input_tensor, tf.float32)
    norm = cast / 255
    return norm
