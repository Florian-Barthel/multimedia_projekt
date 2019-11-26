import nodes


def build(images_placeholder, labels_placeholder, batch_size, num_scales, num_aspect_ratios):
    norm = nodes.normalize(input_tensor=images_placeholder)
    net = nodes.mobile_net_v2(norm)

    convolution = nodes.convolution(input_tensor=net,
                                    scales=num_scales,
                                    aspect_ratios=num_aspect_ratios)

    reshape = nodes.reshape(input_tensor=convolution,
                            batch_size=batch_size,
                            scales=num_scales,
                            aspect_ratios=num_aspect_ratios)

    calculate_loss = nodes.calculate_loss(input_tensor=reshape,
                                          labels_tensor=labels_placeholder)
    return calculate_loss

