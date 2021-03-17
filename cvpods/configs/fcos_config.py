#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # META_ARCHITECTURE="RetinaNet",
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"]),
        FPN=dict(IN_FEATURES=["res3", "res4", "res5"]),
        SHIFT_GENERATOR=dict(
            NUM_SHIFTS=1,
            # Relative offset between the center of the first shift and the top-left corner of img
            # Units: fraction of feature map stride (e.g., 0.5 means half stride)
            # Allowed values are floats in [0, 1) range inclusive.
            # Recommended value is 0.5, although it is not expected to affect model accuracy.
            OFFSET=0.0,
        ),
        FCOS=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            NUM_CONVS=4,
            FPN_STRIDES=[8, 16, 32, 64, 128],
            PRIOR_PROB=0.01,
            CENTERNESS_ON_REG=False,
            NORM_REG_TARGETS=False,
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="iou",
            CENTER_SAMPLING_RADIUS=0.0,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            NORM_SYNC=True,
        ),
    ),
)


class FCOSConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = FCOSConfig()
