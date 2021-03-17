#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import logging
import math
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

from .solo import SOLO, SOLOHead, multi_apply, points_nms


class DecoupledSOLO(SOLO):
    """
    Implement Decoupled SOLO.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_head(self, cfg, feature_shapes):
        assert self.head_type == 'DecoupledSOLOHead'
        self.head = DecoupledSOLOHead(cfg, feature_shapes)

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
            ins_preds_x, ins_preds_y, cate_preds = self.head(features, eval=False)
            featmap_sizes = [featmap.size()[-2:] for featmap in ins_preds_x]
            ins_label_list, cate_label_list, ins_ind_label_list, ins_ind_label_list_xy = \
                self.get_ground_truth(gt_instances, featmap_sizes)
            return self.losses(ins_preds_x, ins_preds_y, cate_preds, ins_label_list,
                               cate_label_list, ins_ind_label_list, ins_ind_label_list_xy)
        else:
            ins_preds_x, ins_preds_y, cate_preds = self.head(features, eval=True)
            results = self.inference(ins_preds_x, ins_preds_y, cate_preds, batched_inputs)
            processed_results = [{"instances": r} for r in results]
            return processed_results

    def losses(self,
               ins_preds_x,
               ins_preds_y,
               cate_preds,
               ins_label_list,
               cate_label_list,
               ins_ind_label_list,
               ins_ind_label_list_xy):
        # ins, per level
        ins_labels = []  # per level
        for ins_labels_level, ins_ind_labels_level in \
                zip(zip(*ins_label_list), zip(*ins_ind_label_list)):
            ins_labels_per_level = []
            for ins_labels_level_img, ins_ind_labels_level_img in \
                    zip(ins_labels_level, ins_ind_labels_level):
                ins_labels_per_level.append(
                    ins_labels_level_img[ins_ind_labels_level_img, ...]
                )
            ins_labels.append(torch.cat(ins_labels_per_level))

        ins_preds_x_final = []
        for ins_preds_level_x, ins_ind_labels_level in \
                zip(ins_preds_x, zip(*ins_ind_label_list_xy)):
            ins_preds_x_final_per_level = []
            for ins_preds_level_img_x, ins_ind_labels_level_img in \
                    zip(ins_preds_level_x, ins_ind_labels_level):
                ins_preds_x_final_per_level.append(
                    ins_preds_level_img_x[ins_ind_labels_level_img[:, 1], ...]
                )
            ins_preds_x_final.append(torch.cat(ins_preds_x_final_per_level))

        ins_preds_y_final = []
        for ins_preds_level_y, ins_ind_labels_level in \
                zip(ins_preds_y, zip(*ins_ind_label_list_xy)):
            ins_preds_y_final_per_level = []
            for ins_preds_level_img_y, ins_ind_labels_level_img in \
                    zip(ins_preds_level_y, ins_ind_labels_level):
                ins_preds_y_final_per_level.append(
                    ins_preds_level_img_y[ins_ind_labels_level_img[:, 0], ...]
                )
            ins_preds_y_final.append(torch.cat(ins_preds_y_final_per_level))

        num_ins = 0.
        # dice loss, per_level
        loss_ins = []
        for input_x, input_y, target in zip(ins_preds_x_final, ins_preds_y_final, ins_labels):
            mask_n = input_x.size(0)
            if mask_n == 0:
                continue
            num_ins += mask_n
            input = (input_x.sigmoid()) * (input_y.sigmoid())
            target = target.float() / 255.
            loss_ins.append(dice_loss(input, target))

        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.loss_ins_weight

        # cate
        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        cate_preds = torch.cat(cate_preds)

        cate_labels = []
        for cate_labels_level in zip(*cate_label_list):
            cate_labels_per_level = []
            for cate_labels_level_img in cate_labels_level:
                cate_labels_per_level.append(
                    cate_labels_level_img.flatten()
                )
            cate_labels.append(torch.cat(cate_labels_per_level))
        flatten_cate_labels = torch.cat(cate_labels)
        foreground_idxs = flatten_cate_labels != self.num_classes
        cate_labels = torch.zeros_like(cate_preds)
        cate_labels[foreground_idxs, flatten_cate_labels[foreground_idxs]] = 1

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
        """
        ins_label_list, cate_label_list, ins_ind_label_list, ins_ind_label_list_xy = multi_apply(
            self.solo_target_single_image,
            gt_instances,
            featmap_sizes=featmap_sizes)
        return ins_label_list, cate_label_list, ins_ind_label_list, ins_ind_label_list_xy

    @torch.no_grad()
    def solo_target_single_image(self, gt_instance, featmap_sizes):
        """
        Prepare ground truth for single image.
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
        ins_ind_label_list_xy = []  # per level
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
                ins_label = torch.zeros([1, featmap_size[0], featmap_size[1]], dtype=torch.uint8,
                                        device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                # default is False
                ins_ind_label = torch.zeros([1], dtype=torch.bool, device=device)
                ins_ind_label_list.append(ins_ind_label)
                ins_ind_label_list_xy.append(torch.zeros([0, 2], dtype=torch.int64, device=device))
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

            # Instance mask
            ins_label = ins_label[ins_ind_label]
            ins_label_list.append(ins_label)
            # Instance category
            cate_label_list.append(cate_label)
            # Instance index
            ins_ind_label = ins_ind_label[ins_ind_label]
            ins_ind_label_list.append(ins_ind_label)
            # Instance coordinate
            foreground_idxs = (cate_label != self.num_classes)
            ins_ind_label_list_xy.append(foreground_idxs.nonzero(as_tuple=False))

        return ins_label_list, cate_label_list, ins_ind_label_list, ins_ind_label_list_xy

    @torch.no_grad()
    def inference(self, ins_preds_x, ins_preds_y, cate_preds, batched_inputs):
        assert len(ins_preds_x) == len(cate_preds)
        num_levels = len(cate_preds)
        featmap_size = ins_preds_x[0].size()[-2:]

        results = []
        for img_id, batched_input in enumerate(batched_inputs):
            cate_pred_list = []
            seg_pred_list_x = []
            seg_pred_list_y = []
            for i in range(num_levels):
                cate_pred_list.append(
                    cate_preds[i][img_id].view(-1, self.num_classes).detach()
                )
                seg_pred_list_x.append(
                    ins_preds_x[i][img_id].detach()
                )
                seg_pred_list_y.append(
                    ins_preds_y[i][img_id].detach()
                )
            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            seg_pred_list_x = torch.cat(seg_pred_list_x, dim=0)
            seg_pred_list_y = torch.cat(seg_pred_list_y, dim=0)

            img_shape = batched_input["instances"].image_size
            ori_shape = (batched_input["height"], batched_input["width"])

            results_per_image = self.inference_single_image(
                cate_pred_list, seg_pred_list_x, seg_pred_list_y,
                featmap_size, img_shape, ori_shape)
            results.append(results_per_image)
        return results

    @torch.no_grad()
    def inference_single_image(self,
                               cate_preds,
                               seg_preds_x,
                               seg_preds_y,
                               featmap_size,
                               img_shape,
                               ori_shape):
        result = Instances(ori_shape)

        # overall info.
        h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # trans trans_diff.
        trans_size = torch.Tensor(self.seg_num_grids).pow(2).cumsum(0).long()
        trans_diff = torch.ones(trans_size[-1].item(), device=self.device).long()
        num_grids = torch.ones(trans_size[-1].item(), device=self.device).long()
        seg_size = torch.Tensor(self.seg_num_grids).cumsum(0).long()
        seg_diff = torch.ones(trans_size[-1].item(), device=self.device).long()
        strides = torch.ones(trans_size[-1].item(), device=self.device)

        n_stage = len(self.seg_num_grids)
        trans_diff[:trans_size[0]] *= 0
        seg_diff[:trans_size[0]] *= 0
        num_grids[:trans_size[0]] *= self.seg_num_grids[0]
        strides[:trans_size[0]] *= self.feature_strides[0]

        for ind_ in range(1, n_stage):
            trans_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= trans_size[ind_ - 1]
            seg_diff[trans_size[ind_ - 1]:trans_size[ind_]] *= seg_size[ind_ - 1]
            num_grids[trans_size[ind_ - 1]:trans_size[ind_]] *= self.seg_num_grids[ind_]
            strides[trans_size[ind_ - 1]:trans_size[ind_]] *= self.feature_strides[ind_]

        # process.
        inds = (cate_preds > self.score_threshold)
        # category scores.
        cate_scores = cate_preds[inds]

        # category labels.
        inds = inds.nonzero(as_tuple=False)
        trans_diff = torch.index_select(trans_diff, dim=0, index=inds[:, 0])
        seg_diff = torch.index_select(seg_diff, dim=0, index=inds[:, 0])
        num_grids = torch.index_select(num_grids, dim=0, index=inds[:, 0])
        strides = torch.index_select(strides, dim=0, index=inds[:, 0])

        y_inds = (inds[:, 0] - trans_diff) // num_grids
        x_inds = (inds[:, 0] - trans_diff) % num_grids
        y_inds += seg_diff
        x_inds += seg_diff

        cate_labels = inds[:, 1]
        seg_masks_soft = seg_preds_x[x_inds, ...] * seg_preds_y[y_inds, ...]
        seg_masks = seg_masks_soft > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return result

        seg_masks_soft = seg_masks_soft[keep, ...]
        seg_masks = seg_masks[keep, ...]
        cate_scores = cate_scores[keep]
        sum_masks = sum_masks[keep]
        cate_labels = cate_labels[keep]

        # mask scoring
        seg_score = (seg_masks_soft * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_score

        if len(cate_scores) == 0:
            return result

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.nms_per_image:
            sort_inds = sort_inds[: self.nms_per_image]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        seg_masks = seg_masks[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        sum_masks = sum_masks[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                 kernel=self.nms_kernel, sigma=self.nms_sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= self.update_threshold
        seg_masks_soft = seg_masks_soft[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_detections_per_image:
            sort_inds = sort_inds[: self.max_detections_per_image]
        seg_masks_soft = seg_masks_soft[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_masks_soft = F.interpolate(seg_masks_soft.unsqueeze(0),
                                       size=upsampled_size_out,
                                       mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_masks_soft,
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


class DecoupledSOLOHead(SOLOHead):
    """
    The head used in SOLO for instance segmentation.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)

    def _init_layers(self):
        ins_convs_x = []
        ins_convs_y = []
        cate_convs = []
        for i in range(self.stacked_convs):
            # Mask branch
            chn = self.in_channels + 1 if i == 0 else self.seg_feat_channels
            ins_convs_x.append(
                nn.Conv2d(chn,
                          self.seg_feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False if self.norm else True)
            )
            ins_convs_y.append(
                nn.Conv2d(chn,
                          self.seg_feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False if self.norm else True)
            )

            if self.norm:
                ins_convs_x.append(get_norm(self.norm, self.seg_feat_channels))
                ins_convs_y.append(get_norm(self.norm, self.seg_feat_channels))

            ins_convs_x.append(nn.ReLU(inplace=True))
            ins_convs_y.append(nn.ReLU(inplace=True))

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

        self.ins_convs_x = nn.Sequential(*ins_convs_x)
        self.ins_convs_y = nn.Sequential(*ins_convs_y)
        self.cate_convs = nn.Sequential(*cate_convs)

        self.solo_ins_list_x = nn.ModuleList()
        self.solo_ins_list_y = nn.ModuleList()
        for seg_num_grid in self.seg_num_grids:
            self.solo_ins_list_x.append(
                nn.Conv2d(
                    self.seg_feat_channels,
                    seg_num_grid,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            )
            self.solo_ins_list_y.append(
                nn.Conv2d(
                    self.seg_feat_channels,
                    seg_num_grid,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            )

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels,
            self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _init_weights(self):
        for modules in [self.ins_convs_x, self.ins_convs_y, self.cate_convs]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for modules in [self.solo_ins_list_x, self.solo_ins_list_y, self.solo_cate]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01, bias=bias_value)

    def forward(self, features, eval=False):
        new_features = self.split_features(features)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_features]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        ins_pred_x, ins_pred_y, cate_pred = multi_apply(
            self.forward_single_level,
            new_features,
            list(range(len(self.seg_num_grids))),
            eval=eval,
            upsampled_size=upsampled_size
        )
        return ins_pred_x, ins_pred_y, cate_pred

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
        ins_feat_x = torch.cat([ins_feat, x], 1)
        ins_feat_y = torch.cat([ins_feat, y], 1)

        ins_feat_x = self.ins_convs_x(ins_feat_x)
        ins_feat_y = self.ins_convs_y(ins_feat_y)

        ins_feat_x = F.interpolate(ins_feat_x, scale_factor=2, mode='bilinear')
        ins_feat_y = F.interpolate(ins_feat_y, scale_factor=2, mode='bilinear')

        ins_pred_x = self.solo_ins_list_x[idx](ins_feat_x)
        ins_pred_y = self.solo_ins_list_y[idx](ins_feat_y)

        # Cate branch
        seg_num_grid = self.seg_num_grids[idx]
        # Align
        cate_feat = F.interpolate(cate_feat, size=seg_num_grid, mode='bilinear')
        cate_feat = self.cate_convs(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            ins_pred_x = F.interpolate(ins_pred_x.sigmoid(), size=upsampled_size, mode='bilinear')
            ins_pred_y = F.interpolate(ins_pred_y.sigmoid(), size=upsampled_size, mode='bilinear')
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return ins_pred_x, ins_pred_y, cate_pred
