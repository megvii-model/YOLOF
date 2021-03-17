#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from cvpods.configs.base_config import BaseConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        PIXEL_MEAN=[0.485, 0.456, 0.406],  # RGB
        PIXEL_STD=[0.229, 0.224, 0.225],
        BACKBONE=dict(FREEZE_AT=-1, ),  # do not freeze
        RESNETS=dict(
            NUM_CLASSES=None,
            DEPTH=None,
            OUT_FEATURES=["linear"],
            NUM_GROUPS=1,
            # Options: FrozenBN, GN, "SyncBN", "BN"
            NORM="BN",
            ACTIVATION=dict(
                NAME="ReLU",
                INPLACE=True,
            ),
            # Whether init last bn weight of each BasicBlock or BottleneckBlock to 0
            ZERO_INIT_RESIDUAL=True,
            WIDTH_PER_GROUP=64,
            # Use True only for the original MSRA ResNet; use False for C2 and Torch models
            STRIDE_IN_1X1=False,
            RES5_DILATION=1,
            RES2_OUT_CHANNELS=256,
            STEM_OUT_CHANNELS=64,

            # Deep Stem
            DEEP_STEM=False,
        ),
    ),
    INPUT=dict(FORMAT="RGB"),
    SOLVER=dict(
        IMS_PER_DEVICE=32,  # defalut: 8 gpus x 32 = 256
    ),
)


class BaseClassificationConfig(BaseConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = BaseClassificationConfig()
