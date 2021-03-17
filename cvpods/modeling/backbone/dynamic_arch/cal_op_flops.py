# Count Operation MFLOPs when fix batch to 1
# @author: yanwei.li


def count_Conv_flop(
    in_h, in_w, in_channel, out_channel,
    kernel_size, is_bias=False, stride=1, groups=1
):
    out_h = in_h // stride
    out_w = in_w // stride
    bias_ops = 1 if is_bias else 0
    kernel_ops = kernel_size[0] * kernel_size[1] * (in_channel // groups)
    delta_ops = (kernel_ops + bias_ops) * out_channel * out_h * out_w
    return delta_ops / 1e6


def count_Linear_flop(in_num, out_num, is_bias):
    weight_ops = in_num * out_num
    bias_ops = out_num if is_bias else 0
    delta_ops = weight_ops + bias_ops
    return delta_ops / 1e6


def count_BN_flop(in_h, in_w, in_channel, is_affine):
    multi_affine = 2 if is_affine else 1
    delta_ops = multi_affine * in_h * in_w * in_channel
    return delta_ops / 1e6


def count_ReLU_flop(in_h, in_w, in_channel):
    delta_ops = in_h * in_w * in_channel
    return delta_ops / 1e6


def count_Pool2d_flop(in_h, in_w, out_channel, kernel_size, stride):
    out_h = in_h // stride
    out_w = in_w // stride
    kernel_ops = kernel_size[0] * kernel_size[1]
    delta_ops = kernel_ops * out_w * out_h * out_channel
    return delta_ops / 1e6


def count_ConvBNReLU_flop(
    in_h, in_w, in_channel, out_channel,
    kernel_size, is_bias=False, stride=1,
    groups=1, is_affine=True
):
    flops = 0.0
    flops += count_Conv_flop(
        in_h, in_w, in_channel, out_channel,
        kernel_size, is_bias, stride, groups
    )
    in_h = in_h // stride
    in_w = in_w // stride
    flops += count_BN_flop(in_h, in_w, out_channel, is_affine)
    flops += count_ReLU_flop(in_h, in_w, out_channel)
    return flops
