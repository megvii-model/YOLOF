#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import numpy as np

import torch.nn as nn

from cvpods.layers import Conv2d, get_norm

from .shufflenet import ShuffleNetV2, ShuffleV2Block


class SNet(ShuffleNetV2):
    def __init__(
        self, in_channels, channels, num_classes=None, dropout=False, out_features=None,
        norm="BN"
    ):
        """
        See: https://arxiv.org/pdf/1903.11752.pdf

        Args:
            num_blocks (int): the number of blocks in this stage.
            in_channels (int): the input channel number.
            channels (int): output channel numbers for stem and every stages.
            num_classes (None or int): if None, will not perform classification.
            dropout (bool): whether to use dropout.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "snet3" ...
                If None, will return the output of the last layer.
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (See cvpods.layer.get_norm for more details).
        """
        super(ShuffleNetV2, self).__init__()
        self.stage_out_channels = channels
        self.num_classes = num_classes

        # ---------------- Stem ---------------------- #
        input_channels = self.stage_out_channels[0]
        self.stem = nn.Sequential(*[
            Conv2d(
                in_channels, input_channels, kernel_size=3,
                stride=2, padding=1, bias=False,
                norm=get_norm(norm, input_channels),
                activation=nn.ReLU(inplace=True),
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ])

        # TODO: use a stem class and property stride
        current_stride = 4
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": input_channels}

        # ---------------- Stages --------------------- #
        self.stage_num_blocks = [4, 8, 4]
        self.stages_and_names = []
        for i in range(len(self.stage_num_blocks)):
            num_blocks = self.stage_num_blocks[i]
            output_channels = self.stage_out_channels[i + 1]
            name = "snet" + str(i + 3)
            block_list = make_stage(num_blocks, input_channels, output_channels, norm)
            current_stride = current_stride * np.prod([block.stride for block in block_list])
            stages = nn.Sequential(*block_list)

            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = output_channels
            self.add_module(name, stages)
            self.stages_and_names.append((stages, name))
            input_channels = output_channels

        if len(self.stage_out_channels) == len(self.stage_num_blocks) + 2:
            name = "snet" + str(len(self.stage_num_blocks) + 2) + "-last"
            last_output_channels = self.stage_out_channels[-1]
            last_conv = Conv2d(
                output_channels, last_output_channels,
                kernel_size=1, bias=False,
                norm=get_norm(norm, last_output_channels),
                activation=nn.ReLU(inplace=True)
            )
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = last_output_channels
            self.add_module(name, last_conv)
            self.stages_and_names.append((last_conv, name))
        # ---------------- Classifer ------------------- #
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = dropout
            if dropout:
                self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(self.stage_out_channels[-1], num_classes, bias=False)
            name = "linear"

        self._out_features = [name] if out_features is None else out_features

        self._initialize_weights()


def make_stage(num_blocks, input_channels, output_channels, norm):
    """
    Create a snet stage by creating many blocks.

    Args:
        num_blocks (int): the number of blocks in this stage.
        input_channels (int): the input channel number.
        output_channels (int): the output channel number.
        norm (str or callable): a callable that takes the number of
            channels and return a `nn.Module`, or a pre-defined string
            (See cvpods.layer.get_norm for more details).

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = []
    blocks.append(ShuffleV2Block(
        input_channels, output_channels, mid_channels=output_channels // 2,
        kernel_size=5, stride=2, norm=norm)
    )
    input_channels = output_channels
    for i in range(num_blocks - 1):
        blocks.append(ShuffleV2Block(
            input_channels // 2, output_channels, mid_channels=output_channels // 2,
            kernel_size=5, stride=1, norm=norm)
        )

    return blocks


def build_snet_backbone(cfg, input_shape):
    """
    Create a SNet instance from config.

    Returns:
        SNet: a :class:`SNet` instance.
    """
    channel_mapper = {
        49: [24, 60, 120, 240, 512],
        146: [24, 132, 264, 528],
        535: [48, 248, 496, 992],
    }
    model_depth = cfg.MODEL.SNET.DEPTH
    output_feautres = cfg.MODEL.SNET.OUT_FEATURES
    num_classes = cfg.MODEL.SNET.NUM_CLASSES
    norm = cfg.MODEL.SNET.NORM

    assert model_depth in channel_mapper, "Depth {} not supported.".format(model_depth)
    channels = channel_mapper[model_depth]

    model = SNet(
        input_shape.channels,
        channels,
        num_classes=num_classes,
        dropout=model_depth == 535,
        out_features=output_feautres,
        norm=norm,
    )
    model.freeze(cfg.MODEL.BACKBONE.FREEZE_AT)
    return model
