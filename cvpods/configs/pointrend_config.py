#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        ROI_HEADS=dict(
            # NAME="PointRendROIHeads",
            IN_FEATURES=["p2", "p3", "p4", "p5"],
        ),
        ROI_BOX_HEAD=dict(
            TRAIN_ON_PRED_BOXES=True,
        ),
        ROI_MASK_HEAD=dict(
            # NAME="CoarseMaskHead",
            # Names of the input feature maps to be used by a coarse mask head.
            IN_FEATURES=["p2"],
            FC_DIM=1024,
            NUM_FC=2,
            # The side size of a coarse mask head prediction.
            OUTPUT_SIDE_RESOLUTION=7,
            # True if point head is used.
            POINT_HEAD_ON=True,
        ),
        POINT_HEAD=dict(
            # Names of the input feature maps to be used by a mask point head.
            IN_FEATURES=["p2"],
            NUM_CLASSES=80,
            FC_DIM=256,
            NUM_FC=3,
            # Number of points sampled during training for a mask point head.
            TRAIN_NUM_POINTS=14 * 14,
            # Oversampling parameter for PointRend point sampling during training.
            # Parameter `k` in the original paper.
            OVERSAMPLE_RATIO=3,
            # Importance sampling parameter for PointRend point sampling during training.
            # Parametr `beta` in the original paper.
            IMPORTANCE_SAMPLE_RATIO=0.75,
            # Number of subdivision steps during inference.
            SUBDIVISION_STEPS=5,
            # Maximum number of points selected at each subdivision step (N).
            SUBDIVISION_NUM_POINTS=28 * 28,
            CLS_AGNOSTIC_MASK=False,
            # If True, then coarse prediction features are used as inout for each layer
            # in PointRend's MLP.
            COARSE_PRED_EACH_LAYER=True,
            # COARSE_SEM_SEG_HEAD_NAME="SemSegFPNHead"
        ),
    ),
    INPUT=dict(
        # PointRend for instance segmenation does not work with "polygon" mask_format
        MASK_FORMAT="bitmask",
    ),
    DATALOADER=dict(FILTER_EMPTY_ANNOTATIONS=False,),
)


class PointRendRCNNFPNConfig(RCNNFPNConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = PointRendRCNNFPNConfig()
