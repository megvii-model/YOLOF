#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import logging
import math
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, batched_nms, cat
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import iou_loss
from cvpods.modeling.meta_arch.fcos import Scale
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n


def positive_bag_loss(logits, mask, gaussian_probs):
    # bag_prob = Mean-max(logits)
    weight = (3 * logits).exp() * gaussian_probs * mask
    w = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    bag_prob = (w * logits).sum(dim=1)
    return F.binary_cross_entropy(bag_prob,
                                  torch.ones_like(bag_prob),
                                  reduction='none')


def negative_bag_loss(logits, gamma):
    return logits**gamma * F.binary_cross_entropy(
        logits, torch.zeros_like(logits), reduction='none')


def normal_distribution(x, mu=0, sigma=1):
    return (-(x - mu)**2 / (2 * sigma**2)).exp()


def normalize(x):
    return (x - x.min() + 1e-12) / (x.max() - x.min() + 1e-12)


class AutoAssign(nn.Module):
    """
    Implement AutoAssign (https://arxiv.org/abs/2007.03496).
    """
    def __init__(self, cfg):
        super(AutoAssign, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.reg_weight = cfg.MODEL.FCOS.REG_WEIGHT
        # Inference parameters:
        self.score_threshold = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = AutoAssignHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        # Matching and loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.mu = nn.Parameter(torch.zeros(80, 2))
        self.sigma = nn.Parameter(torch.ones(80, 2))

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
        box_cls, box_delta, box_center = self.head(features)
        shifts = self.shift_generator(features)

        if self.training:
            return self.losses(shifts, gt_instances, box_cls, box_delta,
                               box_center)
        else:
            results = self.inference(box_cls, box_delta, box_center, shifts,
                                     images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, shifts, gt_instances, box_cls, box_delta, box_center):
        box_cls_flattened = [
            permute_to_N_HWA_K(x, self.num_classes) for x in box_cls
        ]
        box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]
        pred_class_logits = cat(box_cls_flattened, dim=1)
        pred_shift_deltas = cat(box_delta_flattened, dim=1)
        pred_obj_logits = cat(box_center_flattened, dim=1)

        pred_class_probs = pred_class_logits.sigmoid()
        pred_obj_probs = pred_obj_logits.sigmoid()
        pred_box_probs = []
        num_foreground = pred_class_logits.new_zeros(1)
        num_background = pred_class_logits.new_zeros(1)
        positive_losses = []
        gaussian_norm_losses = []

        for shifts_per_image, gt_instances_per_image, \
            pred_class_probs_per_image, pred_shift_deltas_per_image, \
            pred_obj_probs_per_image in zip(
                shifts, gt_instances, pred_class_probs, pred_shift_deltas,
                pred_obj_probs):
            locations = torch.cat(shifts_per_image, dim=0)
            labels = gt_instances_per_image.gt_classes
            gt_boxes = gt_instances_per_image.gt_boxes

            target_shift_deltas = self.shift2box_transform.get_deltas(
                locations, gt_boxes.tensor.unsqueeze(1))
            is_in_boxes = target_shift_deltas.min(dim=-1).values > 0

            foreground_idxs = torch.nonzero(is_in_boxes, as_tuple=True)

            with torch.no_grad():
                # predicted_boxes_per_image: a_{j}^{loc}, shape: [j, 4]
                predicted_boxes_per_image = self.shift2box_transform.apply_deltas(
                    pred_shift_deltas_per_image, locations)
                # gt_pred_iou: IoU_{ij}^{loc}, shape: [i, j]
                gt_pred_iou = pairwise_iou(
                    gt_boxes, Boxes(predicted_boxes_per_image)).max(
                        dim=0, keepdim=True).values.repeat(
                            len(gt_instances_per_image), 1)

                # pred_box_prob_per_image: P{a_{j} \in A_{+}}, shape: [j, c]
                pred_box_prob_per_image = torch.zeros_like(
                    pred_class_probs_per_image)
                box_prob = 1 / (1 - gt_pred_iou[foreground_idxs]).clamp_(1e-12)
                for i in range(len(gt_instances_per_image)):
                    idxs = foreground_idxs[0] == i
                    if idxs.sum() > 0:
                        box_prob[idxs] = normalize(box_prob[idxs])
                pred_box_prob_per_image[foreground_idxs[1],
                                        labels[foreground_idxs[0]]] = box_prob
                pred_box_probs.append(pred_box_prob_per_image)

            normal_probs = []
            for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                gt_shift_deltas = self.shift2box_transform.get_deltas(
                    shifts_i, gt_boxes.tensor.unsqueeze(1))
                distances = (gt_shift_deltas[..., :2] - gt_shift_deltas[..., 2:]) / 2
                normal_probs.append(
                    normal_distribution(distances / stride,
                                        self.mu[labels].unsqueeze(1),
                                        self.sigma[labels].unsqueeze(1)))
            normal_probs = torch.cat(normal_probs, dim=1).prod(dim=-1)

            composed_cls_prob = pred_class_probs_per_image[:, labels] * pred_obj_probs_per_image

            # matched_gt_shift_deltas: P_{ij}^{loc}
            loss_box_reg = iou_loss(pred_shift_deltas_per_image.unsqueeze(0),
                                    target_shift_deltas,
                                    box_mode="ltrb",
                                    loss_type=self.iou_loss_type,
                                    reduction="none") * self.reg_weight
            pred_reg_probs = (-loss_box_reg).exp()

            # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            positive_losses.append(
                positive_bag_loss(composed_cls_prob.transpose(1, 0) * pred_reg_probs,
                                  is_in_boxes.float(), normal_probs))

            num_foreground += len(gt_instances_per_image)
            num_background += normal_probs[foreground_idxs].sum().item()

            gaussian_norm_losses.append(
                len(gt_instances_per_image) / normal_probs[foreground_idxs].sum().clamp_(1e-12))

        if dist.is_initialized():
            dist.all_reduce(num_foreground)
            num_foreground /= dist.get_world_size()
            dist.all_reduce(num_background)
            num_background /= dist.get_world_size()

        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        positive_loss = torch.cat(positive_losses).sum() / max(1, num_foreground)

        # pred_box_probs: P{a_{j} \in A_{+}}
        pred_box_probs = torch.stack(pred_box_probs, dim=0)
        # negative_loss: \sum_{j}{ FL( (1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}) ) } / n||B||
        negative_loss = negative_bag_loss(
            pred_class_probs * pred_obj_probs * (1 - pred_box_probs),
            self.focal_loss_gamma).sum() / max(1, num_background)

        loss_pos = positive_loss * self.focal_loss_alpha
        loss_neg = negative_loss * (1 - self.focal_loss_alpha)
        loss_norm = torch.stack(gaussian_norm_losses).mean() * (1 - self.focal_loss_alpha)

        return {
            "loss_pos": loss_pos,
            "loss_neg": loss_neg,
            "loss_norm": loss_norm,
        }

    def inference(self, box_cls, box_delta, box_center, shifts, images):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`AutoAssignHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(shifts) == len(images)
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, shifts_per_image in enumerate(shifts):
            image_size = images.image_sizes[img_idx]
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_cls
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_delta
            ]
            box_ctr_per_image = [
                box_ctr_per_level[img_idx] for box_ctr_per_level in box_center
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image, box_ctr_per_image,
                shifts_per_image, tuple(image_size))
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta, box_center, shifts,
                               image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, box_ctr_i, shifts_i in zip(
                box_cls, box_delta, box_center, shifts):
            # (HxWxK,)
            box_cls_i = (box_cls_i.sigmoid_() * box_ctr_i.sigmoid_()).flatten()

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

            shift_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[shift_idxs]
            shifts_i = shifts_i[shift_idxs]
            # predict boxes
            predicted_boxes = self.shift2box_transform.apply_deltas(
                box_reg_i, shifts_i)

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


class AutoAssignHead(nn.Module):
    """
    The head used in FCOS for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(AutoAssignHead, self).__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # fmt: on
        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels,
                                   num_classes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.obj_score = nn.Conv2d(in_channels,
                                   1,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        # Initialization
        for modules in [
                self.cls_subnet, self.bbox_subnet, self.cls_score,
                self.bbox_pred, self.obj_score
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        torch.nn.init.constant_(self.bbox_pred.bias, 4.0)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
        """
        logits = []
        bbox_reg = []
        obj_logits = []
        for feature, stride, scale in zip(features, self.fpn_strides,
                                          self.scales):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            obj_logits.append(self.obj_score(bbox_subnet))

            bbox_pred = scale(self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * stride)
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, obj_logits
