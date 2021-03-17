#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import logging
import math
from functools import partial
from typing import List

import numpy as np
from PIL import Image
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.layers import ShapeSpec, get_norm, matrix_nms
from cvpods.modeling.losses import dice_loss, sigmoid_focal_loss_jit
from cvpods.modeling.nn_utils.weight_init import normal_init
from cvpods.structures import BitMasks, ImageList, Instances
from cvpods.utils import log_first_n


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep


class SOLO(nn.Module):
    """
    Implement SOLO: Segmenting Objects by Locations.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.SOLO.NUM_CLASSES
        self.in_features = cfg.MODEL.SOLO.IN_FEATURES
        self.seg_num_grids = cfg.MODEL.SOLO.NUM_GRIDS
        self.head_type = cfg.MODEL.SOLO.HEAD.TYPE
        self.scale_ranges = cfg.MODEL.SOLO.SCALE_RANGES
        self.feature_strides = cfg.MODEL.SOLO.FEATURE_STRIDES
        self.sigma = cfg.MODEL.SOLO.SIGMA
        # Loss parameters:
        # category loss
        self.loss_ins_type = cfg.MODEL.SOLO.LOSS_INS.TYPE
        self.loss_ins_weight = cfg.MODEL.SOLO.LOSS_INS.LOSS_WEIGHT
        # mask loss
        self.loss_cat_type = cfg.MODEL.SOLO.LOSS_CAT.TYPE
        self.loss_cat_weight = cfg.MODEL.SOLO.LOSS_CAT.LOSS_WEIGHT
        self.loss_cat_gamma = cfg.MODEL.SOLO.LOSS_CAT.GAMMA
        self.loss_cat_alpha = cfg.MODEL.SOLO.LOSS_CAT.ALPHA
        # Inference parameters:
        self.score_threshold = cfg.MODEL.SOLO.SCORE_THRESH_TEST
        self.mask_threshold = cfg.MODEL.SOLO.MASK_THRESH_TEST
        self.nms_per_image = cfg.MODEL.SOLO.NMS_PER_IMAGE
        self.nms_kernel = cfg.MODEL.SOLO.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.SOLO.NMS_SIGMA
        self.update_threshold = cfg.MODEL.SOLO.UPDATE_THRESH
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self._init_head(cfg, feature_shapes)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def _init_head(self, cfg, feature_shapes):
        assert self.head_type == "SOLOHead"
        self.head = SOLOHead(cfg, feature_shapes)

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

        if self.training:
            ins_preds, cate_preds = self.head(features, eval=False)
            featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds]
            ins_label_list, cate_label_list, ins_ind_label_list = self.get_ground_truth(
                gt_instances, featmap_sizes)
            return self.losses(
                ins_preds, cate_preds, ins_label_list, cate_label_list, ins_ind_label_list)
        else:
            ins_preds, cate_preds = self.head(features, eval=True)
            results = self.inference(ins_preds, cate_preds, batched_inputs)
            processed_results = [{"instances": r} for r in results]
            return processed_results

    def losses(self,
               ins_preds,
               cate_preds,
               ins_label_list,
               cate_label_list,
               ins_ind_label_list):
        """
        Compute losses:

            L = L_cate + λ * L_mask

        Args:
            ins_preds (list[Tensor]): each element in the list is mask prediction results
                of one level, and the shape of each element is [N, G*G, H, W], where:
                * N is the number of images per mini-batch
                * G is the side length of each level of the grids
                * H and W is the height and width of the predicted mask

            cate_preds (list[Tensor]): each element in the list is category prediction results
                of one level, and the shape of each element is [#N, #C, #G, #G], where:
                * C is the number of classes

            ins_label_list (list[list[Tensor]]): each element in the list is mask ground truth
                of one image, and each element is a list which contains mask tensors per level
                with shape [H, W], where:
                * H and W is the ground truth mask size per level (same as `ins_preds`)

            cate_label_list (list[list[Tensor]]): each element in the list is category ground truth
                of one image, and each element is a list which contains tensors with shape [G, G]
                per level.

            ins_ind_label_list (list[list[Tensor]]):  used to indicate which grids contain objects,
                these grids need to calculate mask loss. Each element in the list is indicator
                of one image, and each element is a list which contains tensors with shape [G*G]
                per level。

        Returns:
            dict[str -> Tensor]: losses.
        """
        # ins, per level
        ins_preds_valid = []
        ins_labels_valid = []
        cate_labels_valid = []
        num_images = len(ins_label_list)
        num_levels = len(ins_label_list[0])
        for level_idx in range(num_levels):
            ins_preds_per_level = []
            ins_labels_per_level = []
            cate_labels_per_level = []
            for img_idx in range(num_images):
                valid_ins_inds = ins_ind_label_list[img_idx][level_idx]
                ins_preds_per_level.append(
                    ins_preds[level_idx][img_idx][valid_ins_inds, ...]
                )
                ins_labels_per_level.append(
                    ins_label_list[img_idx][level_idx][valid_ins_inds, ...]
                )
                cate_labels_per_level.append(
                    cate_label_list[img_idx][level_idx].flatten()
                )
            ins_preds_valid.append(torch.cat(ins_preds_per_level))
            ins_labels_valid.append(torch.cat(ins_labels_per_level))
            cate_labels_valid.append(torch.cat(cate_labels_per_level))

        # dice loss, per_level
        loss_ins = []
        for input, target in zip(ins_preds_valid, ins_labels_valid):
            if input.size()[0] == 0:
                continue
            input = torch.sigmoid(input)
            target = target.float() / 255.
            loss_ins.append(dice_loss(input, target))
        # loss_ins (list[Tensor]): each element's shape is [#Ins, #H*#W]
        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.loss_ins_weight

        # cate
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        cate_preds = torch.cat(cate_preds)

        flatten_cate_labels = torch.cat(cate_labels_valid)
        foreground_idxs = flatten_cate_labels != self.num_classes
        cate_labels = torch.zeros_like(cate_preds)
        cate_labels[foreground_idxs, flatten_cate_labels[foreground_idxs]] = 1
        num_ins = foreground_idxs.sum()

        loss_cate = self.loss_cat_weight * sigmoid_focal_loss_jit(
            cate_preds,
            cate_labels,
            alpha=self.loss_cat_alpha,
            gamma=self.loss_cat_gamma,
            reduction="sum",
        ) / max(1, num_ins)
        return dict(loss_ins=loss_ins, loss_cate=loss_cate)

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, featmap_sizes):
        """
        Args:
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
            featmap_sizes (list[]): a list of #level elements. Each is a
                tuple of #feature level feature map size.

        Returns:
            ins_label_list, cate_label_list, ins_ind_label_list: See: method: `losses`.
        """
        ins_label_list, cate_label_list, ins_ind_label_list = multi_apply(
            self.solo_target_single_image,
            gt_instances,
            featmap_sizes=featmap_sizes)
        return ins_label_list, cate_label_list, ins_ind_label_list

    @torch.no_grad()
    def solo_target_single_image(self, gt_instance, featmap_sizes):
        """
        Prepare ground truth for single image.

        Args:
            gt_instance, featmap_sizes: See: method: `get_ground_truth`.

        Returns:
            ins_label_list, cate_label_list, ins_ind_label_list: See: method: `losses`.
        """
        device = self.device
        gt_bboxes_raw = gt_instance.gt_boxes
        gt_labels_raw = gt_instance.gt_classes
        gt_masks_raw = gt_instance.gt_masks

        # ins
        gt_areas = torch.sqrt(gt_bboxes_raw.area())

        ins_label_list = []  # per level
        cate_label_list = []  # per level
        ins_ind_label_list = []  # per level
        for (lower_bound, upper_bound), stride, featmap_size, num_grid in zip(
                self.scale_ranges, self.feature_strides, featmap_sizes, self.seg_num_grids):
            ins_label = torch.zeros([num_grid ** 2, featmap_size[0],
                                     featmap_size[1]], dtype=torch.uint8, device=device)
            cate_label = torch.full(
                [num_grid, num_grid], self.num_classes, dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            hit_indices = ((gt_areas >= lower_bound) & (
                gt_areas <= upper_bound)).nonzero(as_tuple=False).flatten()

            if len(hit_indices) == 0:
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                continue

            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices]
            # move mask to cpu and convert to ndarray for compute gt ins center
            gt_masks = gt_masks.tensor.to("cpu").numpy()

            half_ws = 0.5 * (gt_bboxes.tensor[:, 2] - gt_bboxes.tensor[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes.tensor[:, 3] - gt_bboxes.tensor[:, 1]) * self.sigma

            output_stride = stride / 2
            # For each mask
            for seg_mask, gt_label, half_h, half_w in zip(gt_masks, gt_labels, half_hs, half_ws):
                if seg_mask.sum() < 10:
                    continue
                # mass center
                upsampled_size = (featmap_sizes[0][0] * 4, featmap_sizes[0][1] * 4)
                center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(
                    num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid))
                )
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(
                    num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid))
                )

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                # ins
                scale = 1. / output_stride
                h, w = seg_mask.shape[-2:]
                new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)

                seg_mask = Image.fromarray(seg_mask)
                seg_mask = seg_mask.resize((new_w, new_h), Image.BILINEAR)
                seg_mask = np.array(seg_mask)
                seg_mask = torch.from_numpy(seg_mask)
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)
                        ins_label[label, :seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_ind_label[label] = True
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
        return ins_label_list, cate_label_list, ins_ind_label_list

    @torch.no_grad()
    def inference(self, seg_preds, cate_preds, batched_inputs):
        """
        Args:
            seg_preds (list[Tensor]): predicted mask results, each element's
                shape is [N, G*G, H, W].
            cate_preds (list[Tensor]): predicted category results, each element's
                shape is [N, C, G, G].
                N, G, H, W: See: method: `losses`.
        Returns:
            results (list[Instance]): predicted results after post-processing.
        """
        assert len(seg_preds) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = seg_preds[0].size()[-2:]

        results = []
        for img_id, batched_input in enumerate(batched_inputs):
            cate_pred_list = []
            seg_pred_list = []
            for i in range(num_levels):
                cate_pred_list.append(
                    cate_preds[i][img_id].view(-1, self.num_classes).detach()
                )
                seg_pred_list.append(
                    seg_preds[i][img_id].detach()
                )
            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list = torch.cat(seg_pred_list, dim=0)

            img_shape = batched_input["instances"].image_size
            ori_shape = (batched_input["height"], batched_input["width"])

            results_per_image = self.inference_single_image(
                cate_pred_list, seg_pred_list, featmap_size, img_shape, ori_shape)
            results.append(results_per_image)
        return results

    @torch.no_grad()
    def inference_single_image(self,
                               cate_preds,
                               seg_preds,
                               featmap_size,
                               img_shape,
                               ori_shape):
        """
        Args:
            cate_preds, seg_preds: see: method: `inference`.
            featmap_size (list[tuple]): feature map size per level.
            img_shape (tuple): the size of the image fed into the model (height and width).
            ori_shape (tuple): original image shape (height and width).

        Returns:
            result (Instances): predicted results of single image after post-processing.
        """
        assert len(cate_preds) == len(seg_preds)
        result = Instances(ori_shape)

        # overall info.
        h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > self.score_threshold)
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return result
        # category labels.
        inds = inds.nonzero(as_tuple=False)
        cate_labels = inds[:, 1]

        # strides.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(
            2).cumsum(0)  # [1600, 2896, 3472, 3728, 3872]
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.feature_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.feature_strides[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return result

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.nms_per_image:
            sort_inds = sort_inds[: self.nms_per_image]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=self.nms_kernel, sigma=self.nms_sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= self.update_threshold
        if keep.sum() == 0:
            return result
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_detections_per_image:
            sort_inds = sort_inds[: self.max_detections_per_image]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape,
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        seg_masks = BitMasks(seg_masks)
        result.pred_masks = seg_masks
        result.pred_boxes = seg_masks.get_bounding_boxes()
        result.scores = cate_scores
        result.pred_classes = cate_labels
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


class SOLOHead(nn.Module):
    """
    The head used in SOLO for instance segmentation.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        self.num_classes = cfg.MODEL.SOLO.NUM_CLASSES
        self.seg_num_grids = cfg.MODEL.SOLO.NUM_GRIDS
        self.in_channels = input_shape[0].channels
        self.seg_feat_channels = cfg.MODEL.SOLO.HEAD.SEG_FEAT_CHANNELS
        self.stacked_convs = cfg.MODEL.SOLO.HEAD.STACKED_CONVS
        self.prior_prob = cfg.MODEL.SOLO.HEAD.PRIOR_PROB
        self.norm = cfg.MODEL.SOLO.HEAD.NORM
        # Initialization
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        ins_convs = []
        cate_convs = []
        for i in range(self.stacked_convs):
            # Mask branch
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            ins_convs.append(
                nn.Conv2d(chn,
                          self.seg_feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False if self.norm else True)
            )
            if self.norm:
                ins_convs.append(get_norm(self.norm, self.seg_feat_channels))
            ins_convs.append(nn.ReLU(inplace=True))

            # Category branch
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            cate_convs.append(
                nn.Conv2d(chn,
                          self.seg_feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False if self.norm else True)
            )
            if self.norm:
                cate_convs.append(get_norm(self.norm, self.seg_feat_channels))
            cate_convs.append(nn.ReLU(inplace=True))

        self.ins_convs = nn.Sequential(*ins_convs)
        self.cate_convs = nn.Sequential(*cate_convs)

        self.solo_ins_list = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.solo_ins_list.append(
                nn.Conv2d(
                    self.seg_feat_channels,
                    seg_num_grid ** 2,
                    kernel_size=1)
            )
        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels,
            self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _init_weights(self):
        for modules in [self.ins_convs, self.cate_convs]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for modules in [self.solo_ins_list, self.solo_cate]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_value)

    def split_features(self, features):
        return (
            F.interpolate(features[0], scale_factor=0.5, mode='bilinear'),
            features[1],
            features[2],
            features[3],
            F.interpolate(features[4], size=features[3].shape[-2:], mode='bilinear')
        )

    def forward(self, features, eval=False):
        new_features = self.split_features(features)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_features]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        ins_pred, cate_pred = multi_apply(
            self.forward_single_level,
            new_features,
            list(range(len(self.seg_num_grids))),
            eval=eval,
            upsampled_size=upsampled_size
        )
        return ins_pred, cate_pred

    def forward_single_level(self, x, idx, eval=False, upsampled_size=None):
        ins_feat = x
        cate_feat = x
        # Ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
        y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_feat = torch.cat([ins_feat, coord_feat], 1)

        ins_feat = self.ins_convs(ins_feat)
        ins_feat = F.interpolate(ins_feat, scale_factor=2, mode='bilinear')
        ins_pred = self.solo_ins_list[idx](ins_feat)

        # Cate branch
        seg_num_grid = self.seg_num_grids[idx]
        # Align
        cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
        cate_feat = self.cate_convs(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            ins_pred = F.interpolate(ins_pred.sigmoid(), size=upsampled_size, mode='bilinear')
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return ins_pred, cate_pred
