#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch.nn as nn
import torch.nn.functional as F

from cvpods.modeling.nn_utils.feature_utils import gather_feature


class reg_l1_loss(nn.Module):

    def __init__(self):
        super(reg_l1_loss, self).__init__()

    def forward(self, output, mask, index, target):
        pred = gather_feature(output, index, use_transform=True)
        mask = mask.unsqueeze(dim=2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss
