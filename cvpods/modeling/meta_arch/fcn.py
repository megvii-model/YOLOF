#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

from typing import Dict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import Conv2d, ConvTranspose2d, ShapeSpec


class FCNHead(nn.Module):
    """
    The head used in FCN for Semantic Segmentation.
    See: https://arxiv.org/abs/1605.06211 for more details.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT

        upsampling_strides = []
        feature_strides_list = list(feature_strides.values())
        upsampling_strides.append(feature_strides_list[0])
        feature_strides_list = feature_strides_list[::-1]
        for s1, s2 in zip(feature_strides_list[:], feature_strides_list[1:]):
            upsampling_strides.append(s1 // s2)
        assert len(upsampling_strides) == len(self.in_features)

        score_convs = []
        upsampling_convs = []
        for idx, in_feature in enumerate(self.in_features):
            ch = feature_channels[in_feature]
            score_convs.append(
                Conv2d(ch, num_classes, kernel_size=1)
            )
            stride = upsampling_strides[idx]
            upsampling_convs.append(
                ConvTranspose2d(
                    num_classes,
                    num_classes,
                    kernel_size=stride * 2,
                    stride=stride,
                    padding=1,
                    bias=False,
                )
            )
        self.score_convs = nn.ModuleList(score_convs)
        self.upsampling_convs = nn.ModuleList(upsampling_convs)
        self._initialize_weights()

    def _initialize_weights(self):
        # Ref: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
        def get_upsampling_weight(in_channels, out_channels, kernel_size):
            """
            Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
            """
            factor = (kernel_size + 1) // 2
            if kernel_size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:kernel_size, :kernel_size]
            filt = (1 - abs(og[0] - center) / factor) * \
                (1 - abs(og[1] - center) / factor)
            weight = np.zeros(
                (in_channels, out_channels, kernel_size, kernel_size),
                dtype=np.float64
            )
            weight[range(in_channels), range(out_channels), :, :] = filt
            return torch.from_numpy(weight).float()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features, ori_shape=targets.shape[-2:])
        if self.training:
            return None, self.losses(x, targets)
        else:
            return x, {}

    def layers(self, features, ori_shape):
        # NOTE The compute order is from back to front
        for i, f in zip(range(-1, -len(features) - 1, -1), self.in_features[::-1]):
            if i == -1:
                x = self.score_convs[i](features[f])
                pre = self.upsampling_convs[i](x)
            else:
                x = self.score_convs[i](features[f])
                # Crop
                h, w = pre.shape[-2:]
                crop_offset_h = (x.size(-2) - pre.size(-2)) // 2
                crop_offset_w = (x.size(-1) - pre.size(-1)) // 2
                cur = x[:, :, crop_offset_h: crop_offset_h + h, crop_offset_w: crop_offset_w + w]
                # Fuse
                x = pre + cur
                pre = self.upsampling_convs[i](x)

        h, w = ori_shape[-2:]
        crop_offset_h = (pre.size(-2) - ori_shape[-2]) // 2
        crop_offset_w = (pre.size(-1) - ori_shape[-1]) // 2
        x = pre[:, :, crop_offset_h: crop_offset_h + h, crop_offset_w: crop_offset_w + w]

        return x

    def losses(self, predictions, targets):
        loss = F.cross_entropy(
            predictions, targets, reduction="mean", ignore_index=self.ignore_value
        )
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses
