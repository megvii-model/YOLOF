#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

# network file -> build basic pipline and decoder for Dynamic Network
from typing import Dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.layers import Conv2d, ShapeSpec, get_norm
from cvpods.modeling.backbone.dynamic_arch import cal_op_flops
from cvpods.modeling.nn_utils import weight_init
from cvpods.modeling.postprocessing import sem_seg_postprocess
from cvpods.structures import ImageList

__all__ = ["DynamicNet4Seg", "SemSegDecoderHead", "BudgetConstraint"]


class DynamicNet4Seg(nn.Module):
    """
    This module implements Dynamic Network for Semantic Segmentation.
    """
    def __init__(self, cfg):
        super().__init__()
        self.constrain_on = cfg.MODEL.BUDGET.CONSTRAIN
        self.unupdate_rate = cfg.MODEL.BUDGET.UNUPDATE_RATE
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = cfg.build_backbone(cfg)
        self.sem_seg_head = SemSegDecoderHead(
            cfg, self.backbone.output_shape())
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            -1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            -1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.budget_constrint = BudgetConstraint(cfg)
        self.iter = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)

        # step_rate: a float, calculated by current_step/total_step,
        #         This parameter is used for Scheduled Drop Path.
        step_rate = self.iter * 1.0 / self.max_iter
        self.iter += 1
        features, expt_flops, real_flops = self.backbone(
            images.tensor, step_rate)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, False,
                self.sem_seg_head.ignore_value).tensor
        else:
            targets = None

        results, losses = self.sem_seg_head(features, targets)
        # calculate flops
        real_flops += self.sem_seg_head.flops
        # remove grad, avoid adding flops to the loss sum
        real_flops = real_flops.detach().requires_grad_(False)
        expt_flops = expt_flops.detach().requires_grad_(False)
        flops = {'real_flops': real_flops, 'expt_flops': expt_flops}
        # use budget constraint for training
        if self.training:
            if self.constrain_on and step_rate >= self.unupdate_rate:
                warm_up_rate = min(
                    1.0, (step_rate - self.unupdate_rate) / 0.02
                )
                loss_budget = self.budget_constrint(
                    expt_flops, warm_up_rate=warm_up_rate
                )
                losses.update({'loss_budget': loss_budget})

            losses.update(flops)
            return losses

        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs,
                                                       images.image_sizes):
            height = input_per_image.get("height")
            width = input_per_image.get("width")
            r = sem_seg_postprocess(result, image_size, height, width)
            processed_results.append({"sem_seg": r, "flops": flops})
        return processed_results


class SemSegDecoderHead(nn.Module):
    """
    This module implements simple decoder head for Semantic Segmentation.
    It creats decoder on top of the dynamic backbone.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides = {k: v.stride for k, v in input_shape.items()}  # noqa:F841
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        feature_resolution = {
            k: np.array([v.height, v.width])
            for k, v in input_shape.items()
        }
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        self.cal_flops = cfg.MODEL.CAL_FLOPS
        self.real_flops = 0.0
        # fmt: on

        self.layer_decoder_list = nn.ModuleList()
        # set affine in BatchNorm
        if 'Sync' in norm:
            affine = True
        else:
            affine = False
        # use simple decoder
        for _feat in self.in_features:
            res_size = feature_resolution[_feat]
            in_channel = feature_channels[_feat]
            if _feat == 'layer_0':
                out_channel = in_channel
            else:
                out_channel = in_channel // 2
            conv_1x1 = Conv2d(in_channel,
                              out_channel,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False,
                              norm=get_norm(norm, out_channel),
                              activation=nn.ReLU())
            self.real_flops += cal_op_flops.count_ConvBNReLU_flop(
                res_size[0],
                res_size[1],
                in_channel,
                out_channel, [1, 1],
                is_affine=affine)
            self.layer_decoder_list.append(conv_1x1)
        # using Kaiming init
        for layer in self.layer_decoder_list:
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in')
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        in_channel = feature_channels['layer_0']
        # the output layer
        self.predictor = Conv2d(in_channels=in_channel,
                                out_channels=num_classes,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.real_flops += cal_op_flops.count_Conv_flop(
            feature_resolution['layer_0'][0], feature_resolution['layer_0'][1],
            in_channel, num_classes, [3, 3])
        # using Kaiming init
        for m in self.predictor.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_init(m, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features, targets=None):
        pred, pred_output = None, None
        for _index in range(len(self.in_features)):
            out_index = len(self.in_features) - _index - 1
            out_feat = features[self.in_features[out_index]]
            if out_index <= 2:
                out_feat = pred + out_feat
            pred = self.layer_decoder_list[out_index](out_feat)
            if out_index > 0:
                pred = F.interpolate(input=pred,
                                     scale_factor=2,
                                     mode='bilinear',
                                     align_corners=False)
            else:
                pred_output = pred
        # pred output
        pred_output = self.predictor(pred_output)
        pred_output = F.interpolate(input=pred_output,
                                    scale_factor=4,
                                    mode='bilinear',
                                    align_corners=False)

        if self.training:
            losses = {}
            losses["loss_sem_seg"] = (
                F.cross_entropy(
                    pred_output, targets, reduction="mean",
                    ignore_index=self.ignore_value
                ) * self.loss_weight
            )
            return [], losses
        else:
            return pred_output, {}

    @property
    def flops(self):
        return self.real_flops


class BudgetConstraint(nn.Module):
    """
    Given budget constraint to reduce expected inference FLOPs in the Dynamic Network.
    """
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        self.loss_weight = cfg.MODEL.BUDGET.LOSS_WEIGHT
        self.loss_mu = cfg.MODEL.BUDGET.LOSS_MU
        self.flops_all = cfg.MODEL.BUDGET.FLOPS_ALL
        self.warm_up = cfg.MODEL.BUDGET.WARM_UP
        # fmt: on

    def forward(self, flops_expt, warm_up_rate=1.0):
        if self.warm_up:
            warm_up_rate = min(1.0, warm_up_rate)
        else:
            warm_up_rate = 1.0
        losses = self.loss_weight * warm_up_rate * (
            (flops_expt / self.flops_all - self.loss_mu)**2
        )
        return losses
