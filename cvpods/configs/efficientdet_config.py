#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=[0.485, 0.456, 0.406],  # mean value from ImageNet
        PIXEL_STD=[0.229, 0.224, 0.225],
        EFFICIENTNET=dict(
            MODEL_NAME="efficientnet-b0",  # default setting for EfficientDet-D0
            NORM="BN",
            BN_MOMENTUM=1 - 0.99,
            BN_EPS=1e-3,
            DROP_CONNECT_RATE=1 - 0.8,  # survival_prob = 0.8
            DEPTH_DIVISOR=8,
            MIN_DEPTH=None,
            NUM_CLASSES=None,
            FIX_HEAD_STEAM=False,
            MEMORY_EFFICIENT_SWISH=True,
            OUT_FEATURES=["stage4", "stage6", "stage8"],
        ),
        BIFPN=dict(
            IN_FEATURES=["stage4", "stage6", "stage8"],
            NORM="BN",
            BN_MOMENTUM=0.01,  # 1 - 0.99
            BN_EPS=1e-3,
            MEMORY_EFFICIENT_SWISH=True,
            INPUT_SIZE=512,  # default setting for EfficientDet-D0
            NUM_LAYERS=3,  # default setting for EfficientDet-D0
            OUT_CHANNELS=60,  # default setting for EfficientDet-D0
            FUSE_TYPE="fast",  # select in ["softmax", "fast", "sum"]
        ),
        EFFICIENTDET=dict(
            IN_FEATURES=[f"p{i}" for i in range(3, 8)],  # p3-p7
            NUM_CLASSES=80,
            FREEZE_BACKBONE=False,
            FREEZE_BN=False,
            HEAD=dict(
                NUM_CONV=3,  # default setting for EfficientDet-D0
                NORM="BN",
                BN_MOMENTUM=1 - 0.99,
                BN_EPS=1e-3,
                PRIOR_PROB=0.01,
                MEMORY_EFFICIENT_SWISH=True,
            ),
            IOU_THRESHOLDS=[0.5, 0.5],
            IOU_LABELS=[0, -1, 1],
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.5,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=1.5,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.1,
            REG_NORM=4.0,
            BOX_LOSS_WEIGHT=50.0,
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[
                [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)]
                for x in [4 * 2**i for i in range(3, 8)]
            ]
        ),
    ),
)


class EfficientDetConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = EfficientDetConfig()
