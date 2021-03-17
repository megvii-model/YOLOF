#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import logging
import math
from typing import List

import torch
from torch import nn

from cvpods.layers import ShapeSpec, batched_nms, cat
from cvpods.modeling.box_regression import Box2BoxTransform
from cvpods.modeling.losses import smooth_l1_loss
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n

from .retinanet import permute_to_N_HWA_K


def positive_bag_loss(logits, dim):
    # bag_prob = Mean-max(logits)
    weight = 1 / (1 - logits)
    weight /= weight.sum(dim).unsqueeze(dim=-1)
    bag_prob = (weight * logits).sum(dim)
    # positive_bag_loss is binary CE loss of (bag_prob, ones_like(bag_prob))
    return -bag_prob.log()


def negative_bag_loss(logits, gamma):
    binary_ce = -(1 - logits).log()
    return logits**gamma * binary_ce


class FreeAnchor(nn.Module):
    """
    Implement RetinaNet (https://arxiv.org/abs/1708.02002).
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
        self.reg_weight = cfg.MODEL.RETINANET.REG_WEIGHT
        # Inference parameters:
        self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RetinaNetHead(cfg, feature_shapes)
        self.anchor_generator = cfg.build_anchor_generator(cfg, feature_shapes)

        # Matching and loss
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS)
        self.pos_anchor_topk = cfg.MODEL.FREE_ANCHOR.POS_ANCHOR_TOPK
        self.bbox_threshold = cfg.MODEL.FREE_ANCHOR.BBOX_THRESHOLD

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
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
            log_first_n(
                logging.WARN,
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
            return self.losses(anchors, gt_instances, box_cls, box_delta)
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

    def losses(self, anchors, gt_instances, box_cls, box_delta):
        anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]

        box_cls_flattened = [
            permute_to_N_HWA_K(x, self.num_classes) for x in box_cls
        ]
        box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        pred_class_logits = cat(box_cls_flattened, dim=1)
        pred_anchor_deltas = cat(box_delta_flattened, dim=1)

        pred_class_probs = pred_class_logits.sigmoid()
        pred_box_probs = []
        num_foreground = 0
        positive_losses = []
        for anchors_per_image, \
            gt_instances_per_image, \
            pred_class_probs_per_image, \
            pred_anchor_deltas_per_image in zip(
                anchors, gt_instances, pred_class_probs, pred_anchor_deltas):
            gt_classes_per_image = gt_instances_per_image.gt_classes

            with torch.no_grad():
                # predicted_boxes_per_image: a_{j}^{loc}, shape: [j, 4]
                predicted_boxes_per_image = self.box2box_transform.apply_deltas(
                    pred_anchor_deltas_per_image, anchors_per_image.tensor)
                # gt_pred_iou: IoU_{ij}^{loc}, shape: [i, j]
                gt_pred_iou = pairwise_iou(gt_instances_per_image.gt_boxes,
                                           Boxes(predicted_boxes_per_image))

                t1 = self.bbox_threshold
                t2 = gt_pred_iou.max(dim=1, keepdim=True).values.clamp_(
                    min=t1 + torch.finfo(torch.float32).eps)
                # gt_pred_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                gt_pred_prob = ((gt_pred_iou - t1) / (t2 - t1)).clamp_(min=0, max=1)

                # pred_box_prob_per_image: P{a_{j} \in A_{+}}, shape: [j, c]
                nonzero_idxs = torch.nonzero(gt_pred_prob, as_tuple=True)
                pred_box_prob_per_image = torch.zeros_like(pred_class_probs_per_image)
                pred_box_prob_per_image[nonzero_idxs[1], gt_classes_per_image[nonzero_idxs[0]]] \
                    = gt_pred_prob[nonzero_idxs]
                pred_box_probs.append(pred_box_prob_per_image)

            # construct bags for objects
            match_quality_matrix = pairwise_iou(
                gt_instances_per_image.gt_boxes, anchors_per_image)
            _, foreground_idxs = torch.topk(match_quality_matrix,
                                            self.pos_anchor_topk,
                                            dim=1,
                                            sorted=False)

            # matched_pred_class_probs_per_image: P_{ij}^{cls}
            matched_pred_class_probs_per_image = torch.gather(
                pred_class_probs_per_image[foreground_idxs], 2,
                gt_classes_per_image.view(-1, 1, 1).repeat(1, self.pos_anchor_topk, 1)
            ).squeeze(2)

            # matched_gt_anchor_deltas_per_image: P_{ij}^{loc}
            matched_gt_anchor_deltas_per_image = self.box2box_transform.get_deltas(
                anchors_per_image.tensor[foreground_idxs],
                gt_instances_per_image.gt_boxes.tensor.unsqueeze(1))
            loss_box_reg = smooth_l1_loss(
                pred_anchor_deltas_per_image[foreground_idxs],
                matched_gt_anchor_deltas_per_image,
                beta=self.smooth_l1_loss_beta,
                reduction="none").sum(dim=-1) * self.reg_weight
            matched_pred_reg_probs_per_image = (-loss_box_reg).exp()

            # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            num_foreground += len(gt_instances_per_image)
            positive_losses.append(
                positive_bag_loss(
                    matched_pred_class_probs_per_image
                    * matched_pred_reg_probs_per_image,
                    dim=1)
            )

        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_foreground)

        # pred_box_probs: P{a_{j} \in A_{+}}
        pred_box_probs = torch.stack(pred_box_probs, dim=0)
        # negative_loss: \sum_{j}{ FL( (1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}) ) } / n||B||
        negative_loss = negative_bag_loss(
            pred_class_probs * (1 - pred_box_probs),
            self.focal_loss_gamma).sum() / max(1, num_foreground * self.pos_anchor_topk)

        loss_pos = positive_loss * self.focal_loss_alpha
        loss_neg = negative_loss * (1 - self.focal_loss_alpha)

        return {"loss_pos": loss_pos, "loss_neg": loss_neg}

    def inference(self, box_cls, box_delta, anchors, images):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (ImageList): the input images

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
        for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta,
                                                   anchors):
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
        keep = batched_nms(boxes_all, scores_all, class_idxs_all,
                           self.nms_threshold)
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
                                        self.backbone.size_divisibility)
        return images


class RetinaNetHead(nn.Module):
    """
    The head used in RetinaNet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = cfg.build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_anchors * num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_anchors * 4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        # Initialization
        for modules in [
                self.cls_subnet, self.bbox_subnet, self.cls_score,
                self.bbox_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg
