#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import logging
import math
from typing import List

import numpy as np

import torch
import torch.nn as nn

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms
from cvpods.layers.deform_conv import DeformConv
from cvpods.modeling.losses import sigmoid_focal_loss_jit, smooth_l1_loss
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import log_first_n


class RepPoints(nn.Module):
    """
    Implement RepPoints (https://arxiv.org/abs/1904.11490)
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        model_params = cfg.MODEL.REPPOINTS
        # pyramid feature:
        self.num_classes = model_params.NUM_CLASSES
        self.in_features = model_params.IN_FEATURES
        self.fpn_strides = model_params.FPN_STRIDES

        # loss parameters:
        self.focal_loss_gamma = model_params.FOCAL_LOSS_GAMMA
        self.focal_loss_alpha = model_params.FOCAL_LOSS_ALPHA
        self.loss_cls_weight = model_params.LOSS_CLS_WEIGHT
        self.loss_bbox_init_weight = model_params.LOSS_BBOX_INIT_WEIGHT
        self.loss_bbox_refine_weight = model_params.LOSS_BBOX_REFINE_WEIGHT

        # reppoints parameters:
        self.point_base_scale = model_params.POINT_BASE_SCALE
        self.num_points = model_params.NUM_POINTS
        self.transform_method = model_params.TRANSFORM_METHOD
        self.moment_mul = model_params.MOMENT_MUL

        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2),
                                                requires_grad=True)

        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = RepPointsHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        # inference parameters:
        self.score_threshold = model_params.SCORE_THRESH_TEST
        self.topk_candidates = model_params.TOPK_CANDIDATES_TEST
        self.nms_threshold = model_params.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of : class:`DatasetMapper`.
            Each item in the list contains the input for one image.
            For now, each item in the list is a dict that contains:
             * images: Tensor, image in (C, H, W) format.
             * instances: Instances.
             Other information that' s included in the original dict ,such as:
             * "height", "width"(int): the output resolution of the model,
             used in inference.See  `postprocess` for detail
        Return:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss, Used
                during training only.
                At inference stage, return predicted bboxes.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(logging.WARN,
                        "'targets' in the model inputs is \
                            now renamed to 'instances'!",
                        n=10)
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        cls_outs, pts_outs_init, pts_outs_refine = self.head(features)
        center_pts = self.shift_generator(features)

        if self.training:
            return self.losses(center_pts, cls_outs, pts_outs_init,
                               pts_outs_refine, gt_instances)
        else:
            results = self.inference(center_pts, cls_outs, pts_outs_init,
                                     pts_outs_refine, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(self, center_pts, cls_outs, pts_outs_init, pts_outs_refine,
               targets):
        """
        Args:
            center_pts: (list[list[Tensor]]): a list of N=#image elements. Each
                is a list of #feature level tensors. The tensors contains
                shifts of this image on the specific feature level.
            cls_outs: List[Tensor], each item in list with
                shape:[N, num_classes, H, W]
            pts_outs_init: List[Tensor], each item in list with
                shape:[N, num_points*2, H, W]
            pts_outs_refine: List[Tensor], each item in list with
            shape:[N, num_points*2, H, W]
            targets: (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.
                Specify `targets` during training only.

        Returns:
            dict[str:Tensor]:
                mapping from a named loss to scalar tensor
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_outs]
        assert len(featmap_sizes) == len(center_pts[0])

        pts_dim = 2 * self.num_points

        cls_outs = [
            cls_out.permute(0, 2, 3, 1).reshape(cls_out.size(0), -1,
                                                self.num_classes)
            for cls_out in cls_outs
        ]
        pts_outs_init = [
            pts_out_init.permute(0, 2, 3, 1).reshape(pts_out_init.size(0), -1,
                                                     pts_dim)
            for pts_out_init in pts_outs_init
        ]
        pts_outs_refine = [
            pts_out_refine.permute(0, 2, 3, 1).reshape(pts_out_refine.size(0),
                                                       -1, pts_dim)
            for pts_out_refine in pts_outs_refine
        ]

        cls_outs = torch.cat(cls_outs, dim=1)
        pts_outs_init = torch.cat(pts_outs_init, dim=1)
        pts_outs_refine = torch.cat(pts_outs_refine, dim=1)

        pts_strides = []
        for i, s in enumerate(center_pts[0]):
            pts_strides.append(
                cls_outs.new_full((s.size(0), ), self.fpn_strides[i]))
        pts_strides = torch.cat(pts_strides, dim=0)

        center_pts = [
            torch.cat(c_pts, dim=0).to(self.device) for c_pts in center_pts
        ]

        pred_cls = []
        pred_init = []
        pred_refine = []

        target_cls = []
        target_init = []
        target_refine = []

        num_pos_init = 0
        num_pos_refine = 0

        for img, (per_center_pts, cls_prob, pts_init, pts_refine,
                  per_targets) in enumerate(
                      zip(center_pts, cls_outs, pts_outs_init, pts_outs_refine,
                          targets)):
            assert per_center_pts.shape[:-1] == cls_prob.shape[:-1]

            gt_bboxes = per_targets.gt_boxes.to(cls_prob.device)
            gt_labels = per_targets.gt_classes.to(cls_prob.device)

            pts_init_bbox_targets, pts_init_labels_targets = \
                self.point_targets(per_center_pts,
                                   pts_strides,
                                   gt_bboxes.tensor,
                                   gt_labels)

            # per_center_pts, shape:[N, 18]
            per_center_pts_repeat = per_center_pts.repeat(1, self.num_points)

            normalize_term = self.point_base_scale * pts_strides
            normalize_term = normalize_term.reshape(-1, 1)

            # bbox_center = torch.cat([per_center_pts, per_center_pts], dim=1)
            per_pts_strides = pts_strides.reshape(-1, 1)
            pts_init_coordinate = pts_init * per_pts_strides + \
                per_center_pts_repeat
            init_bbox_pred = self.pts_to_bbox(pts_init_coordinate)

            foreground_idxs = (pts_init_labels_targets >= 0) & \
                (pts_init_labels_targets != self.num_classes)

            pred_init.append(init_bbox_pred[foreground_idxs]
                             / normalize_term[foreground_idxs])
            target_init.append(pts_init_bbox_targets[foreground_idxs]
                               / normalize_term[foreground_idxs])
            num_pos_init += foreground_idxs.sum()

            # A another way to convert predicted offset to bbox
            # bbox_pred_init = self.pts_to_bbox(pts_init.detach()) * \
            #     per_pts_strides
            # init_bbox_pred = bbox_center + bbox_pred_init

            pts_refine_bbox_targets, pts_refine_labels_targets = \
                self.bbox_targets(init_bbox_pred, gt_bboxes, gt_labels)

            pts_refine_coordinate = pts_refine * per_pts_strides + \
                per_center_pts_repeat
            refine_bbox_pred = self.pts_to_bbox(pts_refine_coordinate)

            # bbox_pred_refine = self.pts_to_bbox(pts_refine) * per_pts_strides
            # refine_bbox_pred = bbox_center + bbox_pred_refine

            foreground_idxs = (pts_refine_labels_targets >= 0) & \
                (pts_refine_labels_targets != self.num_classes)

            pred_refine.append(refine_bbox_pred[foreground_idxs]
                               / normalize_term[foreground_idxs])
            target_refine.append(pts_refine_bbox_targets[foreground_idxs]
                                 / normalize_term[foreground_idxs])
            num_pos_refine += foreground_idxs.sum()

            gt_classes_target = torch.zeros_like(cls_prob)
            gt_classes_target[foreground_idxs,
                              pts_refine_labels_targets[foreground_idxs]] = 1
            pred_cls.append(cls_prob)
            target_cls.append(gt_classes_target)

        pred_cls = torch.cat(pred_cls, dim=0)
        pred_init = torch.cat(pred_init, dim=0)
        pred_refine = torch.cat(pred_refine, dim=0)

        target_cls = torch.cat(target_cls, dim=0)
        target_init = torch.cat(target_init, dim=0)
        target_refine = torch.cat(target_refine, dim=0)

        loss_cls = sigmoid_focal_loss_jit(
            pred_cls,
            target_cls,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum") / max(
                1, num_pos_refine.item()) * self.loss_cls_weight

        loss_pts_init = smooth_l1_loss(
            pred_init, target_init, beta=0.11, reduction='sum') / max(
                1, num_pos_init.item()) * self.loss_bbox_init_weight

        loss_pts_refine = smooth_l1_loss(
            pred_refine, target_refine, beta=0.11, reduction='sum') / max(
                1, num_pos_refine.item()) * self.loss_bbox_refine_weight

        return {
            "loss_cls": loss_cls,
            "loss_pts_init": loss_pts_init,
            "loss_pts_refine": loss_pts_refine
        }

    def pts_to_bbox(self, points):
        """
        Converting the points set into bounding box.

        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_x = points[:, 0::2]
        pts_y = points[:, 1::2]

        if self.transform_method == "minmax":
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_top = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == "partial_minmax":
            bbox_left = pts_x[:, :4].min(dim=1, keepdim=True)[0]
            bbox_right = pts_x[:, :4].max(dim=1, keepdim=True)[0]
            bbox_top = pts_y[:, :4].min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y[:, :4].max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == "moment":
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_std = pts_x.std(dim=1, keepdim=True)
            pts_y_std = pts_y.std(dim=1, keepdim=True)
            moment_transfer = self.moment_transfer * self.moment_mul + \
                self.moment_transfer.detach() * (1 - self.moment_mul)
            moment_transfer_width = moment_transfer[0]
            moment_transfer_height = moment_transfer[1]
            half_width = pts_x_std * moment_transfer_width.exp()
            half_height = pts_y_std * moment_transfer_height.exp()
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ], dim=1)
        else:
            raise ValueError

        return bbox

    @torch.no_grad()
    def point_targets(self, points, pts_strides, gt_bboxes, gt_labels):
        """
        Target assign: point assign. Compute corresponding GT box and classification targets
        for proposals.

        Args:
            points: pred boxes
            pts_strides: boxes stride of current point(box)
            gt_bboxes: gt boxes
            gt_labels: gt labels

        Returns:
            assigned_bboxes, assigned_labels
        """
        if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        points_lvl = torch.log2(pts_strides).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        # assign gt box
        gt_bboxes_ctr_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)

        scale = self.point_base_scale

        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale)
                          + torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            lvl_points = points[lvl_idx, :]
            gt_point = gt_bboxes_ctr_xy[[idx], :]
            gt_wh = gt_bboxes_wh[[idx], :]

            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            min_dist, min_dist_index = torch.topk(points_gt_dist, 1, largest=False)
            min_dist_points_index = points_index[min_dist_index]
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]
            min_dist_points_index = min_dist_points_index[less_than_recorded_index]

            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

        assigned_bboxes = points.new_zeros((num_points, 4))
        assigned_labels = points.new_full((num_points, ), self.num_classes)

        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = (
                gt_labels[assigned_gt_inds[pos_inds] - 1].to(assigned_labels.dtype)
            )
            assigned_bboxes[pos_inds] = gt_bboxes[assigned_gt_inds[pos_inds] - 1]

        return assigned_bboxes, assigned_labels

    @torch.no_grad()
    def bbox_targets(self,
                     candidate_bboxes,
                     gt_bboxes,
                     gt_labels,
                     pos_iou_thr=0.5,
                     neg_iou_thr=0.4,
                     gt_max_matching=True):
        """
        Target assign: MaxIoU assign

        Args:
            candidate_bboxes:
            gt_bboxes:
            gt_labels:
            pos_iou_thr:
            neg_iou_thr:
            gt_max_matching:

        Returns:

        """
        if candidate_bboxes.size(0) == 0 or gt_bboxes.tensor.size(0) == 0:
            raise ValueError('No gt or anchors')

        candidate_bboxes[:, 0].clamp_(min=0)
        candidate_bboxes[:, 1].clamp_(min=0)
        candidate_bboxes[:, 2].clamp_(min=0)
        candidate_bboxes[:, 3].clamp_(min=0)

        num_candidates = candidate_bboxes.size(0)

        overlaps = pairwise_iou(Boxes(candidate_bboxes), gt_bboxes)
        assigned_labels = overlaps.new_full((overlaps.size(0), ),
                                            self.num_classes,
                                            dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=0)

        bg_inds = max_overlaps < neg_iou_thr
        assigned_labels[bg_inds] = self.num_classes

        fg_inds = max_overlaps >= pos_iou_thr
        assigned_labels[fg_inds] = gt_labels[argmax_overlaps[fg_inds]]

        if gt_max_matching:
            fg_inds = torch.nonzero(overlaps == gt_max_overlaps, as_tuple=False)[:, 0]
            assigned_labels[fg_inds] = gt_labels[argmax_overlaps[fg_inds]]

        assigned_bboxes = overlaps.new_zeros((num_candidates, 4))

        fg_inds = (assigned_labels >= 0) & (assigned_labels
                                            != self.num_classes)
        assigned_bboxes[fg_inds] = gt_bboxes.tensor[argmax_overlaps[fg_inds]]

        return assigned_bboxes, assigned_labels

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)

        return images

    def inference(self, center_pts, cls_outs, pts_outs_init, pts_outs_refine,
                  images):
        """
        Argumments:
            cls_outs, pts_outs_init, pts_outs_refine:
                Same as the output of :`RepPointsHead.forward`
            center_pts: (list[list[Tensor]]): a list of N=#image elements. Each
                is a list of #feature level tensors. The tensors contains
                shifts of this image on the specific feature level.
        Returns:
            results (List[Instances]): a list of #images elements
        """
        assert len(center_pts) == len(images)
        results = []
        cls_outs = [
            x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
            for x in cls_outs
        ]
        pts_outs_init = [
            x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_points * 2)
            for x in pts_outs_init
        ]
        pts_outs_refine = [
            x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_points * 2)
            for x in pts_outs_refine
        ]

        pts_strides = []
        for i, s in enumerate(center_pts[0]):
            pts_strides.append(cls_outs[0].new_full((s.size(0), ),
                                                    self.fpn_strides[i]))
        # pts_strides = torch.cat(pts_strides, dim=0)

        for img_idx, center_pts_per_image in enumerate(center_pts):
            image_size = images.image_sizes[img_idx]
            cls_outs_per_img = [
                cls_outs_per_level[img_idx] for cls_outs_per_level in cls_outs
            ]
            pts_outs_refine_per_img = [
                pts_outs_refine_per_level[img_idx]
                for pts_outs_refine_per_level in pts_outs_refine
            ]
            results_per_img = self.inference_single_image(
                cls_outs_per_img, pts_outs_refine_per_img, pts_strides,
                center_pts_per_image, tuple(image_size))
            results.append(results_per_img)
        return results

    def inference_single_image(self, cls_logits, pts_refine, pts_strides,
                               points, image_size):
        """
        Single-image inference. Return bounding-box detection results by
        thresholding on scores and applying non-maximum suppression (NMS).

        Arguemnts:
            cls_logits (list[Tensor]): list of #feature levels. Each entry
                contains tensor of size (H x W, K)
            pts_refine (list[Tensor]): Same shape as 'cls_logits' except that K
                becomes 2 * num_points.
            pts_strides (list(Tensor)): list of #feature levels. Each entry
                contains tensor of size (H x W, )
            points (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the points for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.
        Returns:
            Same as `inference`, but only for one image
        """
        assert len(cls_logits) == len(pts_refine) == len(pts_strides)
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for cls_logits_i, pts_refine_i, points_i, pts_strides_i in zip(
                cls_logits, pts_refine, points, pts_strides):

            bbox_pos_center = torch.cat([points_i, points_i], dim=1)
            bbox_pred = self.pts_to_bbox(pts_refine_i)
            bbox_pred = bbox_pred * pts_strides_i.reshape(-1,
                                                          1) + bbox_pos_center
            bbox_pred[:, 0].clamp_(min=0, max=image_size[1])
            bbox_pred[:, 1].clamp_(min=0, max=image_size[0])
            bbox_pred[:, 2].clamp_(min=0, max=image_size[1])
            bbox_pred[:, 3].clamp_(min=0, max=image_size[0])

            # (HxWxK, )
            point_cls_i = cls_logits_i.flatten().sigmoid_()

            # keep top k scoring indices only
            num_topk = min(self.topk_candidates, point_cls_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = point_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            point_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            predicted_boxes = bbox_pred[point_idxs]

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


class RepPointsHead(nn.Module):
    """
    The head used in RepPoints for object classification and box regression.
    It has two subnets for the two tasks, which is response for classification
    and regression respectively.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        head_params = cfg.MODEL.REPPOINTS
        self.in_channels = input_shape[0].channels
        self.num_classes = head_params.NUM_CLASSES
        self.feat_channels = head_params.FEAT_CHANNELS
        self.point_feat_channels = head_params.POINT_FEAT_CHANNELS
        self.stacked_convs = head_params.STACK_CONVS
        self.norm_mode = head_params.NORM_MODE
        self.num_points = head_params.NUM_POINTS
        self.gradient_mul = head_params.GRADIENT_MUL
        self.prior_prob = head_params.PRIOR_PROB

        self.dcn_kernel = int(np.sqrt(self.num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                nn.Conv2d(chn,
                          self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            if self.norm_mode == 'GN':
                self.cls_convs.append(
                    nn.GroupNorm(32 * self.feat_channels // 256,
                                 self.feat_channels))
            else:
                raise ValueError('The normalization method in reppoints \
                            head should be GN')
            self.cls_convs.append(nn.ReLU(inplace=True))

            self.reg_convs.append(
                nn.Conv2d(chn,
                          self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            if self.norm_mode == 'GN':
                self.reg_convs.append(
                    nn.GroupNorm(32 * self.feat_channels // 256,
                                 self.feat_channels))
            else:
                raise ValueError('The normalization method in reppoints \
                            head should be GN')
            self.reg_convs.append(nn.ReLU(inplace=True))

        point_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.feat_channels,
                                           self.num_classes, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                point_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  point_out_dim, 1, 1, 0)
        self.init_weights()

    def init_weights(self):
        """
        Initialize model weights
        """
        for modules in [
                self.cls_convs, self.reg_convs, self.reppoints_cls_conv,
                self.reppoints_cls_out, self.reppoints_pts_init_conv,
                self.reppoints_pts_init_out, self.reppoints_pts_refine_conv,
                self.reppoints_pts_refine_out
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.reppoints_cls_out.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors
            in high to low resolutions.Each tensor in the list
            correspond to different feature levels.

        Returns:
            cls_outs (list[Tensor]): list of #feature levels.
            Each entry contains tensor of size (H x W, K)
            pts_outs_init (list[Tensor]): list of #feature levels,
            each entry containstensor of size (H x W, num_points * 2)
            pts_outs_refine (list[Tensor]): list of #feature levels,
            each entry contains tensor of size (H x W, num_points * 2)
        """
        dcn_base_offsets = self.dcn_base_offset.type_as(features[0])

        cls_outs = []
        pts_outs_init = []
        pts_outs_refine = []

        for feature in features:
            reg_feat = cls_feat = feature

            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)

            # initialize reppoints
            pts_out_init = self.reppoints_pts_init_out(
                self.relu(self.reppoints_pts_init_conv(reg_feat)))
            pts_outs_init.append(pts_out_init)
            # refine and classify reppoints
            pts_out_init_grad_mul = (1 - self.gradient_mul) * \
                pts_out_init.detach() + self.gradient_mul * pts_out_init
            dcn_offset = pts_out_init_grad_mul - dcn_base_offsets

            cls_outs.append(
                self.reppoints_cls_out(
                    self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset))
                )
            )
            pts_out_refine = self.reppoints_pts_refine_out(
                self.relu(self.reppoints_pts_refine_conv(reg_feat, dcn_offset))
            )
            pts_out_refine = pts_out_refine + pts_out_init.detach()

            pts_outs_refine.append(pts_out_refine)

        return cls_outs, pts_outs_init, pts_outs_refine
