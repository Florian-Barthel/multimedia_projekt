import tensorflow as tf

def mobile_net_v2():
    return tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1.0,
                                                          include_top=False,
                                                          weights='imagenet',
                                                          pooling=None)