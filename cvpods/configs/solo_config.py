#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        MASK_ON=True,
        PIXEL_MEAN=[123.675, 116.28, 103.53],  # RGB FORMAT
        PIXEL_STD=[1.0, 1.0, 1.0],
        RESNETS=dict(
            DEPTH=50,
            OUT_FEATURES=["res2", "res3", "res4", "res5"],
        ),
        FPN=dict(
            IN_FEATURES=["res2", "res3", "res4", "res5"],
            OUT_CHANNELS=256,
        ),
        SOLO=dict(
            NUM_CLASSES=80,
            IN_FEATURES=["p2", "p3", "p4", "p5", "p6"],
            NUM_GRIDS=[40, 36, 24, 16, 12],  # per level
            SCALE_RANGES=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
            FEATURE_STRIDES=[8, 8, 16, 32, 32],
            # Given a gt: (cx, cy, w, h), the center region is controlled by
            # constant scale factors sigma: (cx, cy, sigma*w, sigma*h)
            SIGMA=0.2,
            HEAD=dict(
                TYPE="SOLOHead",  # "SOLOHead", "DecoupledSOLOHead"
                SEG_FEAT_CHANNELS=256,
                STACKED_CONVS=7,
                PRIOR_PROB=0.01,
                NORM="GN",
                # The following two items are useful in the "DecoupledSOLOLightHead"
                USE_DCN_IN_TOWER=False,
                DCN_TYPE=None,
            ),
            # Loss parameters:
            LOSS_INS=dict(
                TYPE='DiceLoss',
                LOSS_WEIGHT=3.0
            ),
            LOSS_CAT=dict(
                TYPE='FocalLoss',
                GAMMA=2.0,
                ALPHA=0.25,
                LOSS_WEIGHT=1.0,
            ),
            # Inference parameters:
            SCORE_THRESH_TEST=0.1,
            MASK_THRESH_TEST=0.5,
            # NMS parameters:
            NMS_PER_IMAGE=500,
            NMS_KERNEL='gaussian',  # gaussian/linear
            NMS_SIGMA=2.0,
            UPDATE_THRESH=0.05,
            DETECTIONS_PER_IMAGE=100,
        ),
    ),
    INPUT=dict(
        # SOLO for instance segmenation does not work with "polygon" mask_format
        MASK_FORMAT="bitmask",
    )
)


class SOLOConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = SOLOConfig()
