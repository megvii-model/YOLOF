#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

from cvpods.layers import ShapeSpec
from cvpods.structures import ImageList


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Classification(nn.Module):
    """
    ImageNet classification module.
    Weights of this model can be used as pretrained weights of any models in cvpods.
    """
    def __init__(self, cfg):
        super(Classification, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.network = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        self.loss_evaluator = nn.CrossEntropyLoss()

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        preds = self.network(images.tensor)["linear"]

        if self.training:
            labels = torch.tensor([gi["category_id"] for gi in batched_inputs]).cuda()
            losses = self.loss_evaluator(preds, labels)
            acc1, acc5 = accuracy(preds, labels, topk=(1, 5))

            return {
                "loss_cls": losses,
                "Acc@1": acc1,
                "Acc@5": acc5,
            }
        else:
            return preds

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(x.div(255)) for x in images]
        images = ImageList.from_tensors(images, self.network.size_divisibility)
        return images
