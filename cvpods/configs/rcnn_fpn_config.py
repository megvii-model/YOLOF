#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .rcnn_config import RCNNConfig

_config_dict = dict(
    MODEL=dict(
        # BACKBONE=dict(NAME='build_resnet_backbone',),
        RESNETS=dict(OUT_FEATURES=["res2", "res3", "res4", "res5"],),
        FPN=dict(IN_FEATURES=["res2", "res3", "res4", "res5"]),
        ANCHOR_GENERATOR=dict(
            SIZES=[[32], [64], [128], [256], [512]], ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
        ),
        RPN=dict(
            IN_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            PRE_NMS_TOPK_TRAIN=2000,
            PRE_NMS_TOPK_TEST=1000,
            POST_NMS_TOPK_TRAIN=1000,
            POST_NMS_TOPK_TEST=1000,
        ),
        ROI_HEADS=dict(
            # NAME: "StandardROIHeads"
            IN_FEATURES=["p2", "p3", "p4", "p5"],
        ),
        ROI_BOX_HEAD=dict(
            # NAME: "FastRCNNConvFCHead"
            NUM_FC=2,
            POOLER_RESOLUTION=7,
        ),
        ROI_MASK_HEAD=dict(
            # NAME: "MaskRCNNConvUpsampleHead"
            NUM_CONV=4,
            POOLER_RESOLUTION=14,
        ),
    ),
)


class RCNNFPNConfig(RCNNConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = RCNNFPNConfig()
