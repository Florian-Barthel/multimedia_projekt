import nodes


def output(images_placeholder, num_scales, num_aspect_ratios, f_rows, f_cols):
    net = nodes.mobile_net_v2()(images_placeholder, training=False)

    convolution = nodes.convolution(input_tensor=net,
                                    scales=num_scales,
                                    aspect_ratios=num_aspect_ratios)

    reshape = nodes.reshape(input_tensor=convolution,
                            scales=num_scales,
                            aspect_ratios=num_aspect_ratios,
                            f_rows=f_rows,
                            f_cols=f_cols)
    return reshape


def loss(input_tensor, labels_placeholder, negative_percentage):
    return nodes.calculate_loss(input_tensor=input_tensor,
                                labels_tensor=labels_placeholder)
