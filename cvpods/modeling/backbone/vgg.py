#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import torch.nn as nn

from cvpods.layers import get_norm
from cvpods.modeling.backbone import Backbone


class VGG(Backbone):
    """
    This module implements VGG.
    See: https://arxiv.org/pdf/1409.1556.pdf for more details.
    """

    def __init__(self, stage_args, num_classes=None, out_features=None, fc_to_conv=False):
        """
        Args:
            stage_args (list[dict[str->int or str]]): the list contains the configuration dict
                corresponding to each stage of the vgg network.
                Each item in the list is a dict that contains:

                * num_blocks: int, the number of conv layer in the stage.
                * in_channels: int, the number of input tensor channels in the stage.
                * out_channels: int, the number of output tensor channels in the stage.
                * norm: str or callable, the normalization to use.
                * pool_args: tuple, contains the pool parameters of the stage,
                        which are kernel_size, stride, pading, ceil_mode.

            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "Conv1_2", "Conv2_2",
                "Conv3_3" or "Conv4_3"...
                If None, will return the output of the last layer.
            fc_to_conv (bool): if True, change FC6, FC7 to conv layer, this is very useful in SSD.
        """
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self._out_features = list()
        self._out_feature_strides = dict()
        self._out_feature_channels = dict()
        self.stages_and_names = list()

        self.layers = list()
        self.feature_idx_to_name = dict()
        for stage_idx, stage_karg in enumerate(stage_args, 1):
            name = "Conv{}_{}".format(stage_idx, stage_karg["num_blocks"])
            stage = self._make_layers(**stage_karg)
            self.layers.extend(stage)

            self.feature_idx_to_name[len(self.layers) - 2] = name
            self._out_feature_strides[name] = 2 ** (stage_idx - 1)
            self._out_feature_channels[name] = stage_karg["out_channels"]

        if fc_to_conv:
            conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
            conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
            self.layers += [conv6,
                            nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

            self.feature_idx_to_name[len(self.layers) - 1] = "Conv7"
            self._out_feature_strides["Conv7"] = 2 ** (5 - 1)
            self._out_feature_channels["Conv7"] = 1024

        self.features = nn.ModuleList(self.layers)

        # for classification
        if self.num_classes is not None:
            if not fc_to_conv:
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, num_classes),
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(1024 * 19 * 19, num_classes),
                )
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)

    def forward(self, x):
        """
        Args:
            x (Tensor): the input tensor.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to VGG feature map tensor
                in high to low resolution order. Noted that only the feature name
                in parameter `out_features` are returned.
        """
        outputs = dict()
        for idx, layer in enumerate(self.features):
            x = layer(x)
            feature_name = self.feature_idx_to_name.get(idx, None)
            if feature_name is not None and feature_name in self._out_features:
                outputs[feature_name] = x

        if self.num_classes is not None:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            if "linear" in self._out_features:
                outputs["linear"] = x

        return outputs

    def _make_layers(self, num_blocks, **kwargs):
        """
        Create a vgg-net stage by creating many blocks(conv layers).

        Args:
            num_blocks (int): the number of conv layer in the stage.
            kwargs: other arguments, see: method:`__init__`.

        Returns:
            list[nn.Module]: a list of block module.
        """
        blocks = list()
        for _ in range(num_blocks):
            conv2d = nn.Conv2d(
                kwargs["in_channels"], kwargs["out_channels"], kernel_size=3, padding=1)
            if kwargs["norm"]:
                blocks += [conv2d, get_norm(kwargs["norm"],
                                            kwargs["out_channels"]), nn.ReLU(inplace=True)]
            else:
                blocks += [conv2d, nn.ReLU(inplace=True)]
            kwargs["in_channels"] = kwargs["out_channels"]
        pool = nn.MaxPool2d(kernel_size=kwargs["pool_args"][0],
                            stride=kwargs["pool_args"][1],
                            padding=kwargs["pool_args"][2],
                            ceil_mode=kwargs["pool_args"][3])
        blocks.append(pool)
        return blocks


def build_ssd_vgg_backbone(cfg, input_shape):
    """
    Create a VGG instance from config.

    Returns:
        VGG: a :class:`VGG` instance.
    """
    in_channels = input_shape.channels
    vgg_arch = cfg.MODEL.VGG.ARCH
    norm = cfg.MODEL.VGG.NORM
    num_classes = cfg.MODEL.VGG.NUM_CLASSES
    out_features = cfg.MODEL.VGG.OUT_FEATURES
    pool_args = cfg.MODEL.VGG.POOL_ARGS
    fc_to_conv = cfg.MODEL.VGG.FC_TO_CONV

    stage_args = []
    num_blocks_per_stage = {
        'A': (1, 1, 2, 2, 2),
        'B': (2, 2, 2, 2, 2),
        'D': (2, 2, 3, 3, 3),
        'E': (2, 2, 4, 4, 4)
    }[vgg_arch]

    for idx, num_blocks in enumerate(num_blocks_per_stage):
        out_channels = 64 * 2 ** idx if idx < 4 else 512
        stage_kargs = {
            "num_blocks": num_blocks,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            # default: kernel_size=2, stride=2, pading=0, ceil_mode=False
            "pool_args": pool_args.get("pool{}".format(idx + 1), (2, 2, 0, False)),
        }
        in_channels = out_channels
        stage_args.append(stage_kargs)

    model = VGG(stage_args, num_classes, out_features, fc_to_conv)

    return model
