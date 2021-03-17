#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torch.autograd import Function

from cvpods import _C


class _PSROIPool(Function):
    @staticmethod
    def forward(ctx, features, rois, output_size, spatial_scale, group_size, output_dim):
        ctx.pooled_width = int(output_size[0])
        ctx.pooled_height = int(output_size[1])
        ctx.spatial_scale = float(spatial_scale)
        ctx.group_size = int(group_size)
        ctx.output_dim = int(output_dim)

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        mapping_channel = torch.zeros(num_rois, ctx.output_dim,
                                      ctx.pooled_height, ctx.pooled_width).int()
        mapping_channel = mapping_channel.to(features.device)
        output = _C.psroi_pooling_forward_cuda(
            features, rois, mapping_channel,
            ctx.pooled_height, ctx.pooled_width,
            ctx.spatial_scale, ctx.group_size, ctx.output_dim
        )
        ctx.output = output
        ctx.mapping_channel = mapping_channel
        ctx.rois = rois
        ctx.feature_size = features.size()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_input = _C.psroi_pooling_backward_cuda(
            grad_output, ctx.rois, ctx.mapping_channel,
            batch_size, num_channels, data_height, data_width,
            ctx.spatial_scale
            # ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale, ctx.output_dim
        )
        return grad_input, None, None, None, None, None


psroi_pool = _PSROIPool.apply


class PSROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale, group_size, output_dim):
        super(PSROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.group_size = group_size
        self.output_dim = output_dim

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return psroi_pool(
            input, rois, self.output_size, self.spatial_scale, self.group_size, self.output_dim
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", group_size=" + str(self.group_size)
        tmpstr += ", output_dim=" + str(self.output_dim)
        tmpstr += ")"
        return tmpstr
