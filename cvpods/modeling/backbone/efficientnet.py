#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import math
import re
from copy import deepcopy
from easydict import EasyDict as edict

import torch
from torch import nn
from torch.nn import functional as F

from cvpods.layers import Conv2dSamePadding as Conv2d
from cvpods.layers import MemoryEfficientSwish, Swish, get_norm
from cvpods.modeling.backbone import Backbone


def round_filters(channels, global_params, skip=False):
    """
    Calculate and round number of channels based on depth multiplier.

    Args:
        channels (int): base number of channels.
        global_params (EasyDict): global args, see: class: `EfficientNet`.
        skip (bool): if True, do nothing and return the base number of channels.

    Returns:
        int: the number of channels calculated based on the depth multiplier.
    """
    multiplier = global_params.width_coefficient
    if skip or not multiplier:
        return channels
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    channels *= multiplier
    min_depth = min_depth or divisor
    new_channels = max(min_depth, int(channels + divisor / 2) // divisor * divisor)
    # prevent rounding by more than 10%
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return int(new_channels)


def round_repeats(repeats, global_params):
    """
    Round number of repeats based on depth multiplier.

    Args:
        repeats (int): the number of `MBConvBlock` int the stage, see: class: `EfficientNet`.
        global_params (EasyDict): global args, see: class: `EfficientNet`.

    Returns:
        int: the number calculated based on the depth coefficient.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """
    Drop connect.

    Args:
        inputs (Tensor): input tensor.
        p (float): between 0 to 1, drop connect rate.
        training (bool): whether it is training phase.
            if False, will skip drop connect op.

    Returns:
        output (Tensor): the result after drop connect.
    """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    """

    def __init__(self, block_args, global_params):
        """
        Args:
            block_args (EasyDict): block args, see: class: `EfficientNet`.
            global_params (EasyDict): global args, see: class: `EfficientNet`.
        """
        super().__init__()
        self._block_args = block_args
        self.has_se = (block_args.se_ratio is not None) and (
            0 < block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        # Expansion phase
        # number of input channels
        inp = block_args.in_channels
        # number of output channels
        oup = block_args.in_channels * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, padding=0, bias=False)
            self._bn0 = get_norm(global_params.norm, out_channels=oup)

        # Depthwise convolution phase
        k = block_args.kernel_size
        s = block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, padding="SAME", bias=False)
        self._bn1 = get_norm(global_params.norm, out_channels=oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(block_args.in_channels * block_args.se_ratio))
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1, padding=0)
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1, padding=0)

        # Output phase
        final_oup = block_args.out_channels
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, padding=0, bias=False)
        self._bn2 = get_norm(global_params.norm, final_oup)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        Args:
            inputs (Tensor): the input tensor.
            drop_connect_rate (float): float, between 0 to 1, drop connect rate.

        Returns:
            x (Tensor): Output of block.
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        in_channels = self._block_args.in_channels
        out_channels = self._block_args.out_channels
        if self.id_skip and self._block_args.stride == 1 and in_channels == out_channels:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            # skip connection
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        """
        Sets swish function as memory efficient or standard.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(Backbone):
    """
    This module implements EfficientNet.
    See: https://arxiv.org/pdf/1905.11946.pdf for more details.
    """

    def __init__(self, in_channels=3, blocks_args=None, global_params=None,
                 out_features=None):
        """
        Args:
            in_channels (int): Number of input image channels.
            blocks_args (list[EasyDict]): a list of EasyDict to construct blocks.
                Each item in the list contains:

                * num_repeat: int, the number of `MBConvBlock` in the stage.
                * in_channels: int, the number of input tensor channels in the stage.
                * out_channels: int, the number of output tensor channels in the stage.
                * kernel_size: int, the kernel size of conv layer in the stage.
                * stride: int or list or tuple, the stride of conv layer in the stage.
                * expand_ratio: int, the channel expansion ratio at expansion phase
                    in `MBConvBlock`.
                * id_skip: bool, if `True`, apply skip connection in `MBConvBlock`
                    when stride is equal to 1 and the input and output channels are equal.
                * se_ratio: float, Squeeze layer channel reduction ratio in SE module,
                    between 0 and 1.

            global_params (namedtuple): a EasyDict contains global params shared between blocks.
                Which contains:

                * norm: str, the normalization to use.
                * bn_momentum: float, the `momentum` parameter of the norm module.
                * bn_eps: float, the `eps` parameter of the norm module.
                * dropout_rate: dropout rate.
                * num_classes: None or int: if None, will not perform classification.
                * width_coefficient: float, coefficient of width.
                * depth_coefficient: float, coefficient of depth.
                * depth_divisor: int, when calculating and rounding the number of channels
                    of each stage according to the depth coefficient, the number of channels
                    must be an integer multiple of "depth_divisor".
                * min_depth: int, the lower bound of the number of channels in each stage.
                * drop_connect_rate: float, between 0 to 1, drop connect rate.
                * image_size: int, input image size.

            out_features (list[str]): name of the layers whose outputs should be returned
                in forward. Can be anything in "stage1", "stage2", ..., "stage8" or "linear".
                If None, will return the output of the last layer.
        """
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block_args must be greater than 0"
        self._size_divisibility = 0
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._out_features = list()
        self._out_feature_strides = dict()
        self._out_feature_channels = dict()
        self.num_classes = global_params.num_classes

        # Stem
        # number of output channels
        out_channels = round_filters(32, global_params, skip=global_params.fix_head_stem)
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding="SAME", bias=False)
        self._bn0 = get_norm(global_params.norm, out_channels=out_channels)

        # Build blocks
        self._blocks = nn.ModuleList([])
        curr_stride = 2
        curr_block_idx = 0
        self.block_idx_to_name = dict()
        for stage_idx, block_args in enumerate(blocks_args):
            # Update block input and output filters based on depth multiplier.
            block_args.update(
                in_channels=round_filters(block_args.in_channels, global_params),
                out_channels=round_filters(block_args.out_channels, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params)
            )

            name = "stage{}".format(stage_idx + 2)
            curr_stride *= block_args.stride
            self._out_feature_strides[name] = curr_stride
            self._out_feature_channels[name] = block_args.out_channels
            curr_block_idx += block_args.num_repeat
            self.block_idx_to_name[curr_block_idx - 1] = name

            # The first block needs to take care of stride and
            # filter size increase.
            self._blocks.append(MBConvBlock(block_args, global_params))
            if block_args.num_repeat > 1:
                next_block_args = deepcopy(block_args)
                next_block_args.update(in_channels=block_args.out_channels, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(next_block_args, global_params))

        # Head
        if self.num_classes is not None:
            in_channels = block_args.out_channels  # output of final block
            out_channels = round_filters(1280, global_params, skip=global_params.fix_head_stem)
            self._conv_head = Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            self._bn1 = get_norm(global_params.norm, out_channels=out_channels)

            # Final linear layers
            self._avg_pooling = nn.AdaptiveAvgPool2d(1)
            self._dropout = nn.Dropout(global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, global_params.num_classes)
            name = "linear"

        self._swish = MemoryEfficientSwish()

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

        # init bn params
        bn_mom = global_params.bn_momentum
        bn_eps = global_params.bn_eps
        if bn_mom is not None and bn_eps is not None:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = bn_mom
                    m.eps = bn_eps

    def set_swish(self, memory_efficient=True):
        """
        Sets swish function as memory efficient (for training) or standard.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            outputs (dict[str->Tensor]):
                mapping from feature map name to feature map tensor in high to low resolution order,
                shape like (N, C, Hi, Wi).
                Noted that only the feature name in parameter `out_features` are returned.
        """
        outputs = dict()
        bs = inputs.size(0)

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            name = self.block_idx_to_name.get(idx, None)
            if name is not None and name in self._out_features:
                outputs[name] = x

        if self.num_classes is not None:
            # Head
            x = self._swish(self._bn1(self._conv_head(x)))
            # Pooling and final linear layer
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @size_divisibility.setter
    def size_divisibility(self, size_divisibility):
        self._size_divisibility = size_divisibility


def check_model_name_is_valid(model_name):
    """
    Validates model name.
    Model name must be one of efficientnet-b0 ~ b7.
    """
    valid_models = ['efficientnet-b' + str(i) for i in range(9)]
    if model_name not in valid_models:
        raise ValueError('model_name should be one of: {}.'.format(', '.join(valid_models)))


def build_efficientnet_backbone(cfg, input_shape):
    """
    Create a EfficientNet instance from config.

    Returns:
        EfficientNet: a :class:`EfficientNet` instance.
    """
    in_channels = input_shape.channels
    model_name = cfg.MODEL.EFFICIENTNET.MODEL_NAME
    norm = cfg.MODEL.EFFICIENTNET.NORM
    bn_momentum = cfg.MODEL.EFFICIENTNET.BN_MOMENTUM
    bn_eps = cfg.MODEL.EFFICIENTNET.BN_EPS
    drop_connect_rate = cfg.MODEL.EFFICIENTNET.DROP_CONNECT_RATE
    depth_divisor = cfg.MODEL.EFFICIENTNET.DEPTH_DIVISOR
    min_depth = cfg.MODEL.EFFICIENTNET.MIN_DEPTH
    num_classes = cfg.MODEL.EFFICIENTNET.NUM_CLASSES
    fix_head_stem = cfg.MODEL.EFFICIENTNET.FIX_HEAD_STEAM
    memory_efficient = cfg.MODEL.EFFICIENTNET.MEMORY_EFFICIENT_SWISH
    out_features = cfg.MODEL.EFFICIENTNET.OUT_FEATURES

    # when norm is "" or None, set norm = "BN"
    if not norm:
        norm = "BN"

    check_model_name_is_valid(model_name)

    width_coefficient, depth_coefficient, image_size, dropout_rate = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }[model_name]

    global_params = edict(
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
        norm=norm,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=depth_divisor,
        min_depth=min_depth,
        image_size=image_size,
        fix_head_stem=fix_head_stem,
    )

    block_args_str = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]

    blocks_args = []
    for block_args_str_i in block_args_str:
        ops = block_args_str_i.split("_")
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        blocks_args_i = edict(
            num_repeat=int(options['r']),
            in_channels=int(options['i']),
            out_channels=int(options['o']),
            kernel_size=int(options['k']),
            stride=int(options['s'][0]),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_args_str_i),
            se_ratio=float(options['se']) if 'se' in options else None
        )
        blocks_args.append(blocks_args_i)
    model = EfficientNet(in_channels, blocks_args, global_params, out_features)
    model.set_swish(memory_efficient)
    return model
