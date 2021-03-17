#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .rcnn_fpn_config import RCNNFPNConfig

_config_dict = dict(
    MODEL=dict(
        SEM_SEG_HEAD=dict(
            # NAME="SemSegFPNHead",
            IN_FEATURES=["p2", "p3", "p4", "p5"],
            # Label in the semantic segmentation ground truth that is ignored,
            # i.e., no loss is calculated for the correposnding pixel.
            IGNORE_VALUE=255,
            # Number of classes in the semantic segmentation head
            NUM_CLASSES=54,
            # Number of channels in the 3x3 convs inside semantic-FPN heads.
            CONVS_DIM=128,
            # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
            COMMON_STRIDE=4,
            # Normalization method for the convolution layers. Options: "" (no norm), "GN".
            NORM="GN",
            LOSS_WEIGHT=0.5,
        ),
        PANOPTIC_FPN=dict(
            # Scaling of all losses from instance detection / segmentation head.
            INSTANCE_LOSS_WEIGHT=1.0,
            # options when combining instance & semantic segmentation outputs
            COMBINE=dict(
                ENABLED=True,
                OVERLAP_THRESH=0.5,
                STUFF_AREA_LIMIT=4096,
                INSTANCES_CONFIDENCE_THRESH=0.5,
            ),
        ),
    )
)


class PanopticSegmentationConfig(RCNNFPNConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = PanopticSegmentationConfig()
