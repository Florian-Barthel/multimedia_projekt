import nodes


def output(input_tensor):
    net = nodes.mobile_net_v2()(input_tensor, training=False)

    convolution = nodes.convolution(input_tensor=net)

    reshape = nodes.reshape(input_tensor=convolution)
    return reshape


def loss(input_tensor, labels_tensor):
    return nodes.calculate_loss(input_tensor=input_tensor,
                                labels_tensor=labels_tensor)
