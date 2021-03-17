#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_config import BaseConfig

_config_dict = dict(
    MODEL=dict(
        LOAD_PROPOSALS=False,
        MASK_ON=False,
        KEYPOINT_ON=False,
        BACKBONE=dict(FREEZE_AT=0,),
        RESNETS=dict(
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
            NORM="nnSyncBN",
            NUM_GROUPS=1,
            WIDTH_PER_GROUP=64,
            STRIDE_IN_1X1=True,
            RES5_DILATION=1,
            RES2_OUT_CHANNELS=256,
            STEM_OUT_CHANNELS=64,
            DEFORM_ON_PER_STAGE=[False, False, False, False],
            DEFORM_MODULATED=False,
            DEFORM_NUM_GROUPS=1,
        ),
        FPN=dict(
            IN_FEATURES=[],
            OUT_CHANNELS=256,
            NORM="",
            FUSE_TYPE="sum",
        ),
        SEM_SEG_HEAD=dict(
            # NAME="SemSegFPNHead",
            IN_FEATURES=[],
            IGNORE_VALUE=255,
            NUM_CLASSES=(),
            CONVS_DIM=256,
            COMMON_STRIDE=(),
            NORM="GN",
            LOSS_WEIGHT=1.0,
        ),
    ),
    DATALOADER=dict(FILTER_EMPTY_ANNOTATIONS=False,),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            NAME="PolyLR",
            POLY_POWER=0.9,
            MAX_ITER=40000,
            WARMUP_ITERS=1000,
            WARMUP_FACTOR=0.001,
            WARMUP_METHOD="linear",
        ),
        OPTIMIZER=dict(BASE_LR=0.01, ),
        IMS_PER_BATCH=16,
        CHECKPOINT_PERIOD=5000,
    ),
    TEST=dict(PRECISE_BN=dict(ENABLED=True), ),
)


class SemanticSegmentationConfig(BaseConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = SemanticSegmentationConfig()
