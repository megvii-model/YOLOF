#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import logging

from torch import nn

from cvpods.layers import batch_norm
from cvpods.utils.distributed import get_world_size

SYNC_BN_MODULE = (
    nn.SyncBatchNorm,
    batch_norm.NaiveSyncBatchNorm,
    batch_norm.NaiveSyncBatchNorm1d,
)


def maybe_convert_module(model):
    if get_world_size() == 1:
        logger = logging.getLogger(__name__)
        logger.warning("SyncBN used with 1GPU, auto convert to BatchNorm")
        model = convert_syncbn(model)

    return model


def convert_syncbn(module):
    model = module

    if isinstance(module, SYNC_BN_MODULE):
        if isinstance(module, batch_norm.NaiveSyncBatchNorm1d):
            model = nn.BatchNorm1d(module.num_features)
        else:
            model = nn.BatchNorm2d(module.num_features)

        if module.affine:
            model.weight.data = module.weight.data.clone().detach()
            model.bias.data = module.bias.data.clone().detach()
        model.running_mean.data = module.running_mean.data
        model.running_var.data = module.running_var.data
        model.eps = module.eps
    else:  # convert syncbn to bn recurrisvely
        for name, child in module.named_children():
            new_child = convert_syncbn(child)
            if new_child is not child:
                model.add_module(name, new_child)

    return model
