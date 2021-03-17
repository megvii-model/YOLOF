#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import logging
import math
from typing import List

import torch
import torch.nn as nn

from cvpods.layers import (
    MemoryEfficientSwish,
    SeparableConvBlock,
    ShapeSpec,
    Swish,
    cat,
    generalized_batched_nms,
    get_norm
)
from cvpods.modeling.box_regression import Box2BoxTransform
from cvpods.modeling.losses import sigmoid_focal_loss_jit, smooth_l1_loss
from cvpods.modeling.matcher import Matcher
from cvpods.modeling.meta_arch.retinanet import (
    permute_all_cls_and_box_to_N_HWA_K_and_concat,
    permute_to_N_HWA_K
)
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n


class EfficientDet(nn.Module):
    """
    Implement EfficientDet(https://arxiv.org/abs/1911.09070).
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.EFFICIENTDET.NUM_CLASSES
        self.in_features = cfg.MODEL.EFFICIENTDET.IN_FEATURES
        self.freeze_bn = cfg.MODEL.EFFICIENTDET.FREEZE_BN
        self.freeze_backbone = cfg.MODEL.EFFICIENTDET.FREEZE_BACKBONE
        self.input_size = cfg.MODEL.BIFPN.INPUT_SIZE
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.EFFICIENTDET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.EFFICIENTDET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.EFFICIENTDET.SMOOTH_L1_LOSS_BETA
        self.box_loss_weight = cfg.MODEL.EFFICIENTDET.BOX_LOSS_WEIGHT
        self.regress_norm = cfg.MODEL.EFFICIENTDET.REG_NORM
        # Inference parameters:
        self.score_threshold = cfg.MODEL.EFFICIENTDET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.EFFICIENTDET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.EFFICIENTDET.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = EfficientDetHead(cfg, feature_shapes)
        self.anchor_generator = cfg.build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.EFFICIENTDET.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.EFFICIENTDET.IOU_THRESHOLDS,
            cfg.MODEL.EFFICIENTDET.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x / 255. - pixel_mean) / pixel_std

        if self.freeze_bn:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()

        if self.freeze_backbone:
            for name, params in self.named_parameters():
                if name.startswith("backbone.bottom_up"):
                    params.requires_grad = False

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(logging.WARN,
                        "'targets' in the model inputs is now renamed to 'instances'!",
                        n=10)
            gt_instances = [
                x["targets"].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_delta = self.head(features)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(
                anchors, gt_instances)
            return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls,
                               box_delta)
        else:
            results = self.inference(box_cls, box_delta, anchors, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits,
               pred_anchor_deltas):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`EfficientDet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits` and `pred_anchor_deltas`, see
                :meth:`EfficientDetHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        pred_class_logits, pred_anchor_deltas = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_anchor_deltas, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        gt_classes = gt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # Classification loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, num_foreground)

        # Regression loss, refer to the official released code.
        # See: https://github.com/google/automl/blob/master/efficientdet/det_model_fn.py
        loss_box_reg = self.box_loss_weight * self.smooth_l1_loss_beta * smooth_l1_loss(
            pred_anchor_deltas[foreground_idxs],
            gt_anchors_deltas[foreground_idxs],
            beta=self.smooth_l1_loss_beta,
            reduction="sum",
        ) / max(1, num_foreground * self.regress_norm)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    @torch.no_grad()
    def get_ground_truth(self, anchors, targets):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        gt_classes = []
        gt_anchors_deltas = []
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
        # list[Tensor(R, 4)], one for each image

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes,
                                                anchors_per_image)
            gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            if has_gt:
                # ground truth box regression
                matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
                gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
                    anchors_per_image.tensor, matched_gt_boxes.tensor
                )

                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Anchors with label 0 are treated as background.
                gt_classes_i[anchor_labels == 0] = self.num_classes
                # Anchors with label -1 are ignored.
                gt_classes_i[anchor_labels == -1] = -1
            else:
                gt_classes_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros_like(
                    anchors_per_image.tensor)

            gt_classes.append(gt_classes_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

    def inference(self, box_cls, box_delta, anchors, images):
        """
        Args:
            box_cls, box_delta: same as the output of :meth:`EfficientDetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (ImageList): the input images.

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, anchors_per_image,
                tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, anchors, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]
            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(
                box_reg_i, anchors_i.tensor)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = generalized_batched_nms(boxes_all, scores_all, class_idxs_all,
                                       self.nms_threshold, nms_type=self.nms_type)
        keep = keep[:self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility,
                                        pad_ref_long=True,
                                        pad_value=0.0)
        return images


class EfficientDetHead(nn.Module):
    """
    The head used in EfficientDet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.EFFICIENTDET.NUM_CLASSES
        norm = cfg.MODEL.EFFICIENTDET.HEAD.NORM
        bn_momentum = cfg.MODEL.EFFICIENTDET.HEAD.BN_MOMENTUM
        bn_eps = cfg.MODEL.EFFICIENTDET.HEAD.BN_EPS
        prior_prob = cfg.MODEL.EFFICIENTDET.HEAD.PRIOR_PROB
        memory_efficient = cfg.MODEL.EFFICIENTDET.HEAD.MEMORY_EFFICIENT_SWISH
        num_conv_layers = cfg.MODEL.EFFICIENTDET.HEAD.NUM_CONV
        num_anchors = cfg.build_anchor_generator(
            cfg, input_shape).num_cell_anchors

        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.prior_prob = prior_prob

        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"

        num_anchors = num_anchors[0]
        self.cls_subnet = nn.ModuleList([])
        self.bbox_subnet = nn.ModuleList([])
        for _ in range(num_conv_layers):
            self.cls_subnet.append(
                SeparableConvBlock(in_channels, in_channels, kernel_size=3, padding="SAME"))
            self.bbox_subnet.append(
                SeparableConvBlock(in_channels, in_channels, kernel_size=3, padding="SAME"))

        num_levels = len(input_shape)
        self.bn_cls_subnet = nn.ModuleList()
        self.bn_bbox_subnet = nn.ModuleList()
        for _ in range(num_levels):
            self.bn_cls_subnet.append(
                nn.ModuleList([
                    get_norm(norm, in_channels)
                    for _ in range(num_conv_layers)
                ])
            )
            self.bn_bbox_subnet.append(
                nn.ModuleList([
                    get_norm(norm, in_channels)
                    for _ in range(num_conv_layers)
                ])
            )

        self.cls_score = SeparableConvBlock(in_channels,
                                            num_anchors * num_classes,
                                            kernel_size=3,
                                            padding="SAME")
        self.bbox_pred = SeparableConvBlock(in_channels,
                                            num_anchors * 4,
                                            kernel_size=3,
                                            padding="SAME")
        self.act = MemoryEfficientSwish() if memory_efficient else Swish()
        self._init_weights()

    def _init_weights(self):
        """
        Weight initialization as per Tensorflow official implementations.
        See: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/init_ops.py
             #L437
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stddev = math.sqrt(1. / max(1., fan_in))
                m.weight.data.normal_(0, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if self.bn_momentum is not None and self.bn_eps is not None:
                    m.momentum = self.bn_momentum
                    m.eps = self.bn_eps
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            #lvl tensors, each has shape (N, AxK, Hi, Wi).
            logits (list[Tensor]):
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            #lvl tensors, each has shape (N, Ax4, Hi, Wi).
            bbox_reg (list[Tensor]):
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        # each level
        for feature_i, bn_cls_level_i, bn_bbox_level_i in zip(
                features, self.bn_cls_subnet, self.bn_bbox_subnet):
            feature_i_cls = feature_i
            feature_i_bbox = feature_i
            for bn_cls_level_i_depth_i, bn_bbox_level_i_depth_i, cls_subnet_i, bbox_subnet_i in zip(
                    bn_cls_level_i, bn_bbox_level_i, self.cls_subnet, self.bbox_subnet):
                feature_i_cls = self.act(
                    bn_cls_level_i_depth_i(cls_subnet_i(feature_i_cls)))
                feature_i_bbox = self.act(
                    bn_bbox_level_i_depth_i(bbox_subnet_i(feature_i_bbox)))
            logits.append(self.cls_score(feature_i_cls))
            bbox_reg.append(self.bbox_pred(feature_i_bbox))

        return logits, bbox_reg
