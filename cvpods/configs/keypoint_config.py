#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        KEYPOINT_ON=True,
        ROI_KEYPOINT_HEAD=dict(
            NAME="KRCNNConvDeconvUpsampleHead",
            POOLER_RESOLUTION=14,
            POOLER_SAMPLING_RATIO=0,
            CONV_DIMS=tuple(512 for _ in range(8)),
            NUM_KEYPOINTS=17,  # 17 is the number of keypoints in COCO
            # Images with too few (or no) keypoints are excluded from training.
            MIN_KEYPOINTS_PER_IMAGE=1,
            # Normalize by the total number of visible keypoints in the minibatch if True.
            # Otherwise, normalize by the total number of keypoints that could ever exist
            # in the minibatch.
            # The keypoint softmax loss is only calculated on visible keypoints.
            # Since the number of visible keypoints can vary significantly between
            # minibatches, this has the effect of up-weighting the importance of
            # minibatches with few visible keypoints. (Imagine the extreme case of
            # only one visible keypoint versus N: in the case of N, each one
            # contributes 1/N to the gradient compared to the single keypoint
            # determining the gradient direction). Instead, we can normalize the
            # loss by the total number of keypoints, if it were the case that all
            # keypoints were visible in a full minibatch. (Returning to the example,
            # this means that the one visible keypoint contributes as much as each
            # of the N keypoints.)
            NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS=True,
            # Multi-task loss weight to use for keypoints
            # Recommended values:
            #   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
            #   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
            LOSS_WEIGHT=1.0,
            # Type of pooling operation applied to the incoming feature map for each RoI
            POOLER_TYPE="ROIAlignV2",
        ),
    )
)


class KeypointConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = KeypointConfig()
