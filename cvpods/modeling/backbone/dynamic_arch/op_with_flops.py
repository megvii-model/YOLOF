# pylint: disable=W0613

# Routinng candidates for dynamic networks with calculated FLOPs,
# modified Search Space in DARTS to have different input and output channels.
# @author: yanwei.li
import torch
import torch.nn as nn

from cvpods.layers import Conv2d, get_norm
from cvpods.modeling.backbone.dynamic_arch import cal_op_flops as flops
from cvpods.modeling.nn_utils import weight_init

OPS = {
    'none': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        Zero(stride=stride),
    'avg_pool_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        AvgPool2d(
            C_in, C_out, 3, stride=stride, padding=1, input_size=input_size
        ),
    'max_pool_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        MaxPool2d(
            C_in, C_out, 3, stride=stride, padding=1, input_size=input_size
        ),
    'skip_connect': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        Identity(C_in, C_out, norm_layer=norm_layer, affine=affine, input_size=input_size)
        if stride == 1 else
        FactorizedReduce(C_in, C_out, norm_layer=norm_layer, affine=affine, input_size=input_size),
    'sep_conv_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        SepConv(
            C_in, C_out, 3, stride, 1, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'sep_conv_5x5': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        SepConv(
            C_in, C_out, 5, stride, 2, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'sep_conv_7x7': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        SepConv(
            C_in, C_out, 7, stride, 3, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'sep_conv_heavy_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        SepConvHeavy(
            C_in, C_out, 3, stride, 1, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'sep_conv_heavy_5x5': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        SepConvHeavy(
            C_in, C_out, 5, stride, 2, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'dil_conv_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        DilConv(
            C_in, C_out, 3, stride, 2, 2, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'dil_conv_5x5': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        DilConv(
            C_in, C_out, 5, stride, 4, 2, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'conv_3x3_basic': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        BasicResBlock(
            C_in, C_out, 3, stride, 1, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'Bottleneck_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        Bottleneck(
            C_in, C_out, 3, stride, 1, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'Bottleneck_5x5': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        Bottleneck(
            C_in, C_out, 5, stride, 2, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'MBConv_3x3': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        MBConv(
            C_in, C_out, 3, stride, 1, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
    'MBConv_5x5': lambda C_in, C_out, stride, norm_layer, affine, input_size:
        MBConv(
            C_in, C_out, 5, stride, 2, norm_layer=norm_layer,
            affine=affine, input_size=input_size
        ),
}


class BasicResBlock(nn.Module):

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding,
        norm_layer, affine=True, input_size=None
    ):
        super(BasicResBlock, self).__init__()
        self.op = Conv2d(
            C_in, C_out, kernel_size, stride=stride, padding=padding,
            bias=False, norm=get_norm(norm_layer, C_out)
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_in, C_out,
            affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_flop(
        self, kernel_size, stride, in_channel,
        out_channel, affine, in_h, in_w
    ):
        cal_flop = flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel, kernel_size, False, stride
        )
        in_h, in_w = in_h // stride, in_w // stride
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(
        self, C_in, C_out, kernel_size, stride, padding,
        dilation, norm_layer, affine=True, input_size=None
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            Conv2d(
                C_in, C_out, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_out)
            )
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_in, C_out,
            affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_flop(self, kernel_size, stride, in_channel, out_channel, affine, in_h, in_w):
        cal_flop = flops.count_Conv_flop(
            in_h, in_w, in_channel, in_channel, kernel_size,
            False, stride, groups=in_channel
        )
        in_h, in_w = in_h // stride, in_w // stride
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel,
            kernel_size=[1, 1], is_bias=False
        )
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding,
        norm_layer, affine=True, input_size=None
    ):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # depth wise
            Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=C_in, bias=False
            ),
            # point wise
            Conv2d(
                C_in, C_in, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_in),
                activation=nn.ReLU()
            ),
            # stack 2 separate depthwise-conv.
            Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, groups=C_in, bias=False
            ),
            Conv2d(
                C_in, C_out, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_out)
            )
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_in, C_out,
            affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_flop(self, kernel_size, stride, in_channel, out_channel, affine, in_h, in_w):
        cal_flop = flops.count_Conv_flop(
            in_h, in_w, in_channel, in_channel, kernel_size,
            False, stride, groups=in_channel
        )
        in_h, in_w = in_h // stride, in_w // stride
        cal_flop += flops.count_ConvBNReLU_flop(
            in_h, in_w, in_channel, in_channel,
            kernel_size=[1, 1], is_bias=False,
            is_affine=affine
        )
        # stack 2 separate depthwise-conv.
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, in_channel,
            kernel_size, False, stride=1, groups=in_channel
        )
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel,
            kernel_size=[1, 1], is_bias=False
        )
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        return self.op(x)


# Heavy SepConv, stack 3 SepConv
class SepConvHeavy(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding,
        norm_layer, affine=True, input_size=None
    ):
        super(SepConvHeavy, self).__init__()
        self.op = nn.Sequential(
            # depth wise
            Conv2d(
                C_in, C_in, kernel_size=kernel_size,
                stride=stride, padding=padding,
                groups=C_in, bias=False
            ),
            # point wise
            Conv2d(
                C_in, C_in, kernel_size=1, padding=0,
                bias=False, norm=get_norm(norm_layer, C_in),
                activation=nn.ReLU()
            ),
            # stack 2 separate depthwise-conv.
            Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, groups=C_in, bias=False
            ),
            Conv2d(
                C_in, C_in, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_in),
                activation=nn.ReLU()
            ),
            # stack 3 separate depthwise-conv.
            Conv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, groups=C_in, bias=False
            ),
            Conv2d(
                C_in, C_out, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_out)
            )
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_in, C_out,
            affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_flop(self, kernel_size, stride, in_channel, out_channel, affine, in_h, in_w):
        cal_flop = flops.count_Conv_flop(
            in_h, in_w, in_channel, in_channel, kernel_size,
            False, stride, groups=in_channel
        )
        in_h, in_w = in_h // stride, in_w // stride
        cal_flop += flops.count_ConvBNReLU_flop(
            in_h, in_w, in_channel, in_channel,
            kernel_size=[1, 1], is_bias=False, is_affine=affine
        )
        # stack 2 separate depthwise-conv.
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, in_channel,
            kernel_size, False, stride=1, groups=in_channel
        )
        cal_flop += flops.count_ConvBNReLU_flop(
            in_h, in_w, in_channel, in_channel,
            kernel_size=[1, 1], is_bias=False, is_affine=affine
        )
        # stack 3 separate depthwise-conv.
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, in_channel,
            kernel_size, False, stride=1, groups=in_channel
        )
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel,
            kernel_size=[1, 1], is_bias=False
        )
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        return self.op(x)


# using Bottleneck from ResNet
class Bottleneck(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride, padding,
        norm_layer, expansion=4, affine=True, input_size=None
    ):
        super(Bottleneck, self).__init__()
        self.hidden_dim = C_in // expansion
        self.op = nn.Sequential(
            Conv2d(
                C_in, self.hidden_dim, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, self.hidden_dim),
                activation=nn.ReLU()
            ),
            Conv2d(
                self.hidden_dim, self.hidden_dim, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False,
                norm=get_norm(norm_layer, self.hidden_dim),
                activation=nn.ReLU()
            ),
            Conv2d(
                self.hidden_dim, C_out, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_out)
            )
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_in,
            C_out, affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_flop(self, kernel_size, stride, in_channel, out_channel, affine, in_h, in_w):
        cal_flop = flops.count_ConvBNReLU_flop(
            in_h, in_w, in_channel, self.hidden_dim,
            [1, 1], False, is_affine=affine
        )
        cal_flop += flops.count_ConvBNReLU_flop(
            in_h, in_w, self.hidden_dim, self.hidden_dim,
            kernel_size, False, stride=stride, is_affine=affine
        )
        in_h, in_w = in_h // stride, in_w // stride
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, self.hidden_dim, out_channel,
            kernel_size=[1, 1], is_bias=False
        )
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        return self.op(x)


# using MBConv from MobileNet V2
class MBConv(nn.Module):
    def __init__(
        self, C_in, C_out, kernel_size, stride,
        padding, norm_layer, expansion=4,
        affine=True, input_size=None
    ):
        super(MBConv, self).__init__()
        self.hidden_dim = expansion * C_in
        self.op = nn.Sequential(
            # pw
            Conv2d(
                C_in, self.hidden_dim, 1, 1, 0, bias=False,
                norm=get_norm(norm_layer, self.hidden_dim),
                activation=nn.ReLU()
            ),
            # dw
            Conv2d(
                self.hidden_dim, self.hidden_dim, kernel_size,
                stride, padding, groups=self.hidden_dim, bias=False,
                norm=get_norm(norm_layer, self.hidden_dim),
                activation=nn.ReLU()
            ),
            # pw-linear without ReLU!
            Conv2d(
                self.hidden_dim, C_out, 1, 1, 0, bias=False,
                norm=get_norm(norm_layer, C_out)
            )
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_in, C_out,
            affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for m in self.op.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_flop(
        self, kernel_size, stride, in_channel,
        out_channel, affine, in_h, in_w
    ):
        cal_flop = flops.count_ConvBNReLU_flop(
            in_h, in_w, in_channel, self.hidden_dim,
            kernel_size=[1, 1], is_bias=False, is_affine=affine
        )
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, self.hidden_dim, self.hidden_dim, kernel_size,
            False, stride, groups=self.hidden_dim, is_affine=affine
        )
        in_h, in_w = in_h // stride, in_w // stride
        # pw-linear without ReLU!
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, self.hidden_dim, out_channel,
            kernel_size=[1, 1], is_bias=False
        )
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(
        self, C_in, C_out, norm_layer, affine=True, input_size=None
    ):
        super(Identity, self).__init__()
        if C_in == C_out:
            self.change = False
            self.flops = 0.0
        else:
            self.change = True
            self.op = Conv2d(
                C_in, C_out, kernel_size=1, padding=0, bias=False,
                norm=get_norm(norm_layer, C_out)
            )
            self.flops = self.get_flop(
                [1, 1], 1, C_in, C_out, affine, input_size[0], input_size[1]
            )
            # using Kaiming init
            for m in self.op.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in')
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not self.change:
            return x
        else:
            return self.op(x)

    def get_flop(self, kernel_size, stride, in_channel, out_channel, affine, in_h, in_w):
        cal_flop = flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel,
            kernel_size=kernel_size, is_bias=False
        )
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop


class Zero(nn.Module):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride
        self.flops = 0.0

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        else:
            return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, norm_layer, affine=True, input_size=None):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = norm_layer(C_out, affine=affine)

        self.flops = self.get_flop(
            [1, 1], 2, C_in, C_out, affine, input_size[0], input_size[1]
        )
        # using Kaiming init
        for layer in [self.conv_1, self.conv_2]:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in')
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def get_flop(
        self, kernel_size, stride, in_channel,
        out_channel, affine, in_h, in_w
    ):
        cal_flop = flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel // 2,
            kernel_size, False, stride=stride
        )
        cal_flop += flops.count_Conv_flop(
            in_h, in_w, in_channel, out_channel // 2,
            kernel_size, False, stride=stride
        )
        in_h, in_w = in_h // stride, in_w // stride
        cal_flop += flops.count_BN_flop(in_h, in_w, out_channel, affine)
        return cal_flop

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class AvgPool2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, input_size):
        super(AvgPool2d, self).__init__()
        self.avg_pool = nn.AvgPool2d(
            kernel_size, stride=stride, padding=padding, count_include_pad=False
        )
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride,
            C_out, input_size[0], input_size[1]
        )

    def get_flop(self, kernel_size, stride, out_channel, in_h, in_w):
        cal_flop = flops.count_Pool2d_flop(
            in_h, in_w, out_channel, kernel_size, stride
        )
        return cal_flop

    def forward(self, x):
        return self.avg_pool(x)


class MaxPool2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, input_size):
        super(MaxPool2d, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.flops = self.get_flop(
            [kernel_size, kernel_size], stride, C_out,
            input_size[0], input_size[1]
        )

    def get_flop(self, kernel_size, stride, out_channel, in_h, in_w):
        cal_flop = flops.count_Pool2d_flop(
            in_h, in_w, out_channel, kernel_size, stride
        )
        return cal_flop

    def forward(self, x):
        return self.max_pool(x)
