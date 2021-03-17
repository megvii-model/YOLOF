#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # Backbone NAME: "build_retinanet_resnet_fpn_backbone"
        RESNETS=dict(OUT_FEATURES=["res3", "res4", "res5"]),
        FPN=dict(
            IN_FEATURES=["res3", "res4", "res5"],
            BLOCK_IN_FEATURES="res5",
        ),
        ANCHOR_GENERATOR=dict(
            SIZES=[
                [x, x * 2 ** (1.0 / 3), x * 2 ** (2.0 / 3)]
                for x in [32, 64, 128, 256, 512]
            ]
        ),
        RETINANET=dict(
            # This is the number of foreground classes.
            NUM_CLASSES=80,
            IN_FEATURES=["p3", "p4", "p5", "p6", "p7"],
            # Convolutions to use in the cls and bbox tower
            # NOTE: this doesn't include the last conv for logits
            NUM_CONVS=4,
            # IoU overlap ratio [bg, fg] for labeling anchors.
            # Anchors with < bg are labeled negative (0)
            # Anchors  with >= bg and < fg are ignored (-1)
            # Anchors with >= fg are labeled positive (1)
            IOU_THRESHOLDS=[0.4, 0.5],
            IOU_LABELS=[0, -1, 1],
            # Prior prob for rare case (i.e. foreground) at the beginning of training.
            # This is used to set the bias for the logits layer of the classifier subnet.
            # This improves training stability in the case of heavy class imbalance.
            PRIOR_PROB=0.01,
            # Inference cls score threshold, only anchors with score > INFERENCE_TH are
            # considered for inference (to improve speed)
            SCORE_THRESH_TEST=0.05,
            TOPK_CANDIDATES_TEST=1000,
            NMS_THRESH_TEST=0.5,
            # Weights on (dx, dy, dw, dh) for normalizing Retinanet anchor regression targets
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            # Loss parameters
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            SMOOTH_L1_LOSS_BETA=0.1,
        ),
    ),
)


class RetinaNetConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = RetinaNetConfig()
