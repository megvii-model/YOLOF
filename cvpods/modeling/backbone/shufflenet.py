#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import numpy as np

import torch
import torch.nn as nn

from cvpods.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec, get_norm
from cvpods.modeling.backbone import Backbone


class ShuffleV2Block(nn.Module):

    def __init__(
        self, input_channels, output_channels, mid_channels,
        kernel_size, stride, bias=False, norm="BN"
    ):
        """
        Args:
            input_channels (int): the input channel number.
            output_channels (int): the output channel number.
            mid_channels (int): the middle channel number.
            kernel_size (int): the kernel size in conv filters.
            stride (int): the stride of the current block.
            bias (bool): whether to have bias in conv.
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (See cvpods.layer.get_norm for more details).
        """
        super(ShuffleV2Block, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        padding = kernel_size // 2

        delta_channels = output_channels - input_channels
        branch_main = [
            # point-wise conv
            Conv2d(
                input_channels, mid_channels, kernel_size=1, bias=bias,
                norm=get_norm(norm, mid_channels),
                activation=nn.ReLU(inplace=True),
            ),
            # depth-wise conv
            Conv2d(
                mid_channels, mid_channels, kernel_size, stride,
                padding, groups=mid_channels, bias=bias,
                norm=get_norm(norm, mid_channels),
            ),
            # point-wise conv
            Conv2d(
                mid_channels, delta_channels, kernel_size=1, bias=bias,
                norm=get_norm(norm, delta_channels),
                activation=nn.ReLU(inplace=True),
            )
        ]
        self.branch_main = nn.Sequential(*branch_main)

        self.branch_proj = None

        if stride == 2:
            branch_proj = [
                # depth-wise conv
                Conv2d(
                    input_channels, input_channels, kernel_size, stride,
                    padding, groups=input_channels, bias=bias,
                    norm=get_norm(norm, input_channels)
                ),
                # point-wise conv
                Conv2d(
                    input_channels, input_channels, kernel_size=1, bias=bias,
                    norm=get_norm(norm, input_channels),
                    activation=nn.ReLU(inplace=True)
                )
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, x):
        if self.branch_proj is None:
            x_proj, x = self.channel_shuffle(x)
        else:
            x_proj = self.branch_proj(x)

        x = self.branch_main(x)
        return torch.cat([x_proj, x], dim=1)

    def channel_shuffle(self, x):
        N, C, H, W = x.shape
        assert C % 2 == 0, "number of channels must be divided by 2, got {}".format(C)
        # (N, C, H, W) -> (N, C/2, 2, H, W) -> (2, N, C/2, H, W)
        x = x.view(N, C // 2, 2, H, W).permute(2, 0, 1, 3, 4).contiguous()
        return x[0], x[1]

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class ShuffleNetV2(Backbone):

    def __init__(
        self, in_channels, channels, num_classes=None, dropout=False, out_features=None,
        norm="BN"
    ):
        """
        See: https://arxiv.org/pdf/1807.11164.pdf

        Args:
            num_blocks (int): the number of blocks in this stage.
            in_channels (int): the input channel number.
            channels (int): output channel numbers for stem and every stages.
            num_classes (None or int): if None, will not perform classification.
            dropout (bool): whether to use dropout.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "shuffle3" ...
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
            name = "shuffle" + str(i + 3)
            block_list = make_stage(num_blocks, input_channels, output_channels, norm)
            current_stride = current_stride * np.prod([block.stride for block in block_list])
            stages = nn.Sequential(*block_list)

            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = output_channels
            self.add_module(name, stages)
            self.stages_and_names.append((stages, name))
            input_channels = output_channels

        name = "shuffle" + str(len(self.stage_num_blocks) + 2) + "-last"
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
            if self.dropout:
                x = self.dropout(x)
            x = x.reshape(-1, self.stage_out_channels[-1])
            x = self.classifier(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at):
        """
        Args:
            freeze_at (int): freeze the stem and the first `freeze_at - 1` stages.
        """
        if freeze_at >= 1:
            for p in self.stem.parameters():
                p.requires_grad = False
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.stem)

        for i in range(freeze_at - 1):
            FrozenBatchNorm2d.convert_frozen_batchnorm(self.stages_and_names[i][0])

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
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


def make_stage(num_blocks, input_channels, output_channels, norm):
    """
    Create a shufflenetv2 stage by creating many blocks.

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
        kernel_size=3, stride=2, norm=norm)
    )
    input_channels = output_channels
    for i in range(num_blocks - 1):
        blocks.append(ShuffleV2Block(
            input_channels // 2, output_channels, mid_channels=output_channels // 2,
            kernel_size=3, stride=1, norm=norm)
        )

    return blocks


def build_shufflenetv2_backbone(cfg, input_shape):
    """
    Create a ShuffleNetV2 instance from config.

    Returns:
        ShuffleNetV2: a :class:`ShuffleNetV2` instance.
    """
    channel_mapper = {
        "0.5x": [24, 48, 96, 192, 1024],
        "1.0x": [24, 116, 232, 464, 1024],
        "1.5x": [24, 176, 352, 704, 1024],
        "2.0x": [24, 244, 488, 976, 2048],
    }
    model_size = cfg.MODEL.SHUFFLENET.MODEL_SIZE
    output_feautres = cfg.MODEL.SHUFFLENET.OUT_FEATURES
    num_classes = cfg.MODEL.SHUFFLENET.NUM_CLASSES
    norm = cfg.MODEL.SHUFFLENET.NORM

    assert model_size in channel_mapper, "Model size {} not supported.".format(model_size)
    channels = channel_mapper[model_size]

    model = ShuffleNetV2(
        input_shape.channels,
        channels,
        num_classes=num_classes,
        dropout=model_size == "2.0",
        out_features=output_feautres,
        norm=norm,
    )
    model.freeze(cfg.MODEL.BACKBONE.FREEZE_AT)
    return model
