#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy

import numpy as np

import torch.nn as nn

from cvpods.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec, get_activation, get_norm
from cvpods.modeling.backbone import Backbone

__all__ = [
    "InvertedResBlock",
    "MobileStem",
    "MobileNetV2",
    "build_mobilenetv2_backbone",
]


class MobileStem(nn.Module):
    def __init__(self, input_channels, output_channels, norm, activation):
        """
        Args:
            input_channels (int): the input channel number.
            output_channels (int): the output channel number.
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
            activation (str): a pre-defined string
                (See cvpods.layer.get_activation for more details).
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = 2

        self.conv = Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False,
                           norm=get_norm(norm, output_channels),
                           activation=get_activation(activation))

    def forward(self, x):
        return self.conv(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class InvertedResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expand_ratio,
                 norm, activation, use_shortcut=True):
        """
        Args:
            input_channels (int): the input channel number.
            output_channels (int): the output channel number.
            stride (int): the stride of the current block.
            expand_ratio(int): the channel expansion ratio for `mid_channels` in InvertedResBlock.
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (See cvpods.layer.get_norm for more details).
            activation (str): a pre-defined string
                (See cvpods.layer.get_activation for more details).
            use_shortcut (bool): whether to use the residual path.
        """
        super(InvertedResBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        mid_channels = int(round(input_channels * expand_ratio))
        self.use_shortcut = use_shortcut

        if self.use_shortcut:
            assert stride == 1
            assert input_channels == output_channels

        conv_kwargs = {
            "norm": get_norm(norm, mid_channels),
            "activation": get_activation(activation)
        }

        layers = []
        if expand_ratio > 1:
            layers.append(
                Conv2d(input_channels, mid_channels, 1, bias=False,  # Pixel-wise non-linear
                       **deepcopy(conv_kwargs))
            )

        layers += [
            Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False,  # Depth-wise 3x3
                   stride=stride, groups=mid_channels, **deepcopy(conv_kwargs)),
            Conv2d(mid_channels, output_channels, 1, bias=False,  # Pixel-wise linear
                   norm=get_norm(norm, output_channels))
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(Backbone):
    def __init__(
        self,
        stem,
        inverted_residual_setting,
        norm,
        activation,
        num_classes=None,
        out_features=None,
    ):
        """
        See: https://arxiv.org/pdf/1801.04381.pdf

        Args:
            stem (nn.Module): a stem module
            inverted_residual_setting(list of list): Network structure.
                (See https://arxiv.org/pdf/1801.04381.pdf Table 2)
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (See cvpods.layer.get_norm for more details).
            activation (str): a pre-defined string
                (See cvpods.layer.get_activation for more details).
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "MobileNetV23" ...
                If None, will return the output of the last layer.
        """
        super(MobileNetV2, self).__init__()

        self.num_classes = num_classes

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be a "
                             "4-element list, got {}".format(inverted_residual_setting))

        self.stem = stem
        self.last_channel = 1280

        input_channels = stem.output_channels

        current_stride = stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": input_channels}

        # ---------------- Stages --------------------- #
        ext = 0
        self.stages_and_names = []
        for i, (t, c, n, s) in enumerate(inverted_residual_setting):
            # t: expand ratio
            # c: output channels
            # n: block number
            # s: stride
            # See https://arxiv.org/pdf/1801.04381.pdf Table 2 for more details
            if s == 1 and i > 0:
                ext += 1
            else:
                ext = 0

            current_stride *= s
            assert int(np.log2(current_stride)) == np.log2(current_stride)

            name = "mobile" + str(int(np.log2(current_stride)))
            if ext != 0:
                name += "-{}".format(ext + 1)

            stage = nn.Sequential(*make_stage(n, input_channels, c, s, t, norm, activation))

            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = c

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            input_channels = c

        name = "mobile" + str(int(np.log2(current_stride))) + "-last"
        stage = Conv2d(input_channels, self.last_channel, kernel_size=1, bias=False,
                       norm=get_norm("BN", self.last_channel),
                       activation=get_activation(activation))
        self.stages_and_names.append((stage, name))
        self.add_module(name, stage)

        self._out_feature_strides[name] = current_stride
        self._out_feature_channels[name] = self.last_channel

        # ---------------- Classifer ------------------- #
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(self.last_channel, num_classes)
            name = "linear"

        self._out_features = [name] if out_features is None else out_features

        self._initialize_weights()

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stages, name in self.stages_and_names:
            x = stages(x)
            if name in self._out_features:
                outputs[name] = x

        if self.num_classes is not None:
            x = self.avgpool(x)
            x = self.dropout(x)
            x = x.reshape(-1, self.last_channel)
            x = self.classifier(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            ) if name != 'linear' else
            ShapeSpec(
                channels=self.num_classes, height=1
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at):
        if freeze_at >= 1:
            self.stem.freeze()
        for i, (stage, _) in enumerate(self.stages_and_names):
            if (i + 2) > freeze_at:
                break
            for p in stage.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(stage)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_stage(num_blocks, input_channels, output_channels, stride, expand_ratio, norm, activation):
    """
    Create a mobilenetv2 stage by creating many blocks.

    Args:
        num_blocks (int): the number of blocks in this stage.
        input_channels (int): the input channel number.
        output_channels (int): the output channel number.
        stride (int): the stride of the first block. The other blocks will have stride=1.
            A `stride` argument will be passed to the block constructor.
        expand_ratio(int): the channel expansion ratio for `mid_channels` in InvertedResBlock.
        norm (str or callable): a callable that takes the number of
            channels and return a `nn.Module`, or a pre-defined string
            (See cvpods.layer.get_norm for more details).
        activation (str): a pre-defined string
            (See cvpods.layer.get_activation for more details).

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    blocks.append(
        InvertedResBlock(input_channels, output_channels, stride=stride, expand_ratio=expand_ratio,
                         norm=norm, activation=activation, use_shortcut=False)
    )
    for i in range(num_blocks - 1):
        blocks.append(
            InvertedResBlock(output_channels, output_channels, stride=1, expand_ratio=expand_ratio,
                             norm=norm, activation=activation)
        )

    return blocks


def build_mobilenetv2_backbone(cfg, input_shape):
    """
    Create a MobileNetV2 instance from config.

    Returns:
        MobileNetV2: a :class:`MobileNetV2` instance.
    """
    stem = MobileStem(
        input_shape.channels,
        cfg.MODEL.MOBILENET.STEM_OUT_CHANNELS,
        cfg.MODEL.MOBILENET.NORM,
        cfg.MODEL.MOBILENET.ACTIVATION
    )

    model = MobileNetV2(
        stem,
        cfg.MODEL.MOBILENET.INVERTED_RESIDUAL_SETTING,
        cfg.MODEL.MOBILENET.NORM,
        cfg.MODEL.MOBILENET.ACTIVATION,
        cfg.MODEL.MOBILENET.NUM_CLASSES,
        cfg.MODEL.MOBILENET.OUT_FEATURES,
    )

    model.freeze(cfg.MODEL.BACKBONE.FREEZE_AT)
    return model
