#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        # META_ARCHITECTURE='GeneralizedRCNN',
        LOAD_PROPOSALS=False,
        MASK_ON=False,
        KEYPOINT_ON=False,
        ANCHOR_GENERATOR=dict(
            SIZES=[[32, 64, 128, 256, 512]], ASPECT_RATIOS=[[0.5, 1.0, 2.0]],
        ),
        PROPOSAL_GENERATOR=dict(
            # Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
            NAME="RPN",
            MIN_SIZE=0,
        ),
        RPN=dict(
            # HEAD_NAME="StandardRPNHead",
            # Names of the input feature maps to be used by RPN
            # e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
            IN_FEATURES=["res4"],
            # Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
            # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
            BOUNDARY_THRESH=-1,
            # IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
            # Minimum overlap required between an anchor and ground-truth box for the
            # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
            # ==> positive RPN example: 1)
            # Maximum overlap allowed between an anchor and ground-truth box for the
            # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
            # ==> negative RPN example: 0)
            # Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
            # are ignored (-1)
            IOU_THRESHOLDS=[0.3, 0.7],
            IOU_LABELS=[0, -1, 1],
            # Total number of RPN examples per image
            BATCH_SIZE_PER_IMAGE=256,
            # Target fraction of foreground (positive) examples per RPN minibatch
            POSITIVE_FRACTION=0.5,
            # Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
            SMOOTH_L1_BETA=0.0,
            LOSS_WEIGHT=1.0,
            # Number of top scoring RPN proposals to keep before applying NMS
            # When FPN is used, this is *per FPN level* (not total)
            PRE_NMS_TOPK_TRAIN=12000,
            PRE_NMS_TOPK_TEST=6000,
            # Number of top scoring RPN proposals to keep after applying NMS
            # When FPN is used, this limit is applied per level and then again to the union
            # of proposals from all levels
            # NOTE: When FPN is used, the meaning of this config is different from Detectron1.
            # It means per-batch topk in Detectron1, but per-image topk here.
            # See "modeling/rpn/rpn_outputs.py" for details.
            POST_NMS_TOPK_TRAIN=2000,
            POST_NMS_TOPK_TEST=1000,
            # NMS threshold used on RPN proposals
            NMS_THRESH=0.7,
            # NMS type for RPN
            # Format: str. (e.g., 'normal' means using normal nms)
            # Allowed values are 'normal', 'softnms-linear', 'softnms-gaussian'
            NMS_TYPE='normal'
        ),
        ROI_HEADS=dict(
            # ROI_HEADS type: "Res5ROIHeads",
            # Names of the input feature maps to be used by ROI heads
            # Currently all heads (box, mask, ...) use the same input feature map list
            # e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
            IN_FEATURES=["res4"],
            # Number of foreground classes
            NUM_CLASSES=80,
            # IOU overlap ratios [IOU_THRESHOLD]
            # Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
            # Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
            IOU_THRESHOLDS=[0.5],
            IOU_LABELS=[0, 1],
            # RoI minibatch size *per image* (number of regions of interest [ROIs])
            # Total number of RoIs per training minibatch =
            #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
            # E.g., a common configuration is: 512 * 16 = 8192
            BATCH_SIZE_PER_IMAGE=512,
            # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
            POSITIVE_FRACTION=0.25,

            # Only used in test mode

            # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
            # balance obtaining high recall with not having too many low precision
            # detections that will slow down inference post processing steps (like NMS)
            # A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
            # inference.
            SCORE_THRESH_TEST=0.05,
            # Overlap threshold used for non-maximum suppression (suppress boxes with
            # IoU >= this threshold)
            NMS_THRESH_TEST=0.5,
            # If True, augment proposals with ground-truth boxes before sampling proposals to
            # train ROI heads.
            PROPOSAL_APPEND_GT=True,
        ),
        ROI_BOX_HEAD=dict(
            # C4 don't use head name option
            # Options for non-C4 models: FastRCNNConvFCHead,

            # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
            # These are empirically chosen to approximately lead to unit variance targets
            BBOX_REG_WEIGHTS=(10.0, 10.0, 5.0, 5.0),
            # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
            SMOOTH_L1_BETA=0.0,
            POOLER_RESOLUTION=14,
            POOLER_SAMPLING_RATIO=0,
            # Type of pooling operation applied to the incoming feature map for each RoI
            POOLER_TYPE="ROIAlignV2",
            NUM_FC=0,
            # Hidden layer dimension for FC layers in the RoI box head
            FC_DIM=1024,
            NUM_CONV=0,
            # Channel dimension for Conv layers in the RoI box head
            CONV_DIM=256,
            # Normalization method for the convolution layers.
            # Options: "" (no norm), "GN", "SyncBN".
            NORM="",
            # Whether to use class agnostic for bbox regression
            CLS_AGNOSTIC_BBOX_REG=False,
            # If true, RoI heads use bounding boxes predicted by the box head
            # rather than proposal boxes
            TRAIN_ON_PRED_BOXES=False,
        ),
        ROI_BOX_CASCADE_HEAD=dict(
            # The number of cascade stages is implicitly defined by
            # the length of the following two configs.
            BBOX_REG_WEIGHTS=(
                (10.0, 10.0, 5.0, 5.0),
                (20.0, 20.0, 10.0, 10.0),
                (30.0, 30.0, 15.0, 15.0),
            ),
            IOUS=(0.5, 0.6, 0.7),
        ),
        ROI_MASK_HEAD=dict(
            # NAME="MaskRCNNConvUpsampleHead",
            POOLER_RESOLUTION=14,
            POOLER_SAMPLING_RATIO=0,
            # The number of convs in the mask head
            NUM_CONV=0,
            CONV_DIM=256,
            # Normalization method for the convolution layers.
            # Options: "" (no norm), "GN", "SyncBN".
            NORM="",
            # Whether to use class agnostic for mask prediction
            CLS_AGNOSTIC_MASK=False,
            # Type of pooling operation applied to the incoming feature map for each RoI
            POOLER_TYPE="ROIAlignV2",
        ),
    ),
)


class RCNNConfig(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = RCNNConfig()
