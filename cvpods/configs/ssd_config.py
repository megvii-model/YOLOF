#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=[123.675, 116.28, 103.53],  # RGB FORMAT
        PIXEL_STD=[1.0, 1.0, 1.0],
        VGG=dict(
            ARCH='D',
            NORM="",
            NUM_CLASSES=None,
            OUT_FEATURES=["Conv4_3", "Conv7"],
            POOL_ARGS=dict(
                pool3=(2, 2, 0, True),  # k, s, p, ceil_model
                pool5=(3, 1, 1, False)  # k, s, p, ceil_model
            ),
            FC_TO_CONV=True,
        ),
        SSD=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["Conv4_3", "Conv7"],
            EXTRA_LAYER_ARCH={
                # the number after "S" and "S" to denote conv layer with stride=2
                "300": [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
                "512": [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 256],
            },
            IOU_THRESHOLDS=[0.5, 0.5],
            IOU_LABELS=[0, -1, 1],
            BBOX_REG_WEIGHTS=(10.0, 10.0, 5.0, 5.0),
            L2NORM_SCALE=20.0,
            # Loss parameters:
            LOSS_ALPHA=1.0,
            SMOOTH_L1_LOSS_BETA=1.0,
            NEGATIVE_POSITIVE_RATIO=3.0,
            # Inference parameters:
            SCORE_THRESH_TEST=0.02,
            NMS_THRESH_TEST=0.45,
        ),
    )
)


class SSDConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = SSDConfig()
