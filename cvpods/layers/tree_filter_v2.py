#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tree_filter_core import MinimumSpanningTree, RandomSpanningTree, TreeFilter2D


class TreeFilterV2(nn.Module):
    def __init__(self, guide_channels, in_channels, embed_channels, num_groups=1, eps=1e-8):
        super(TreeFilterV2, self).__init__()
        ''' Hyper Parameters '''
        self.eps            = eps
        self.guide_channels = guide_channels
        self.in_channels    = in_channels
        self.embed_channels = embed_channels
        self.num_groups     = num_groups

        ''' Embedding Layers '''
        self.embed_layer = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
        self.conf_layer  = nn.Conv2d(in_channels, num_groups, kernel_size=1, bias=False)
        self.guide_layer = nn.Conv2d(guide_channels, self.embed_channels, kernel_size=1, bias=False)
        self.beta        = nn.Parameter(torch.zeros(num_groups))
        self.gamma       = nn.Parameter(torch.zeros(1))

        '''Core of Tree Filter'''
        self.rst_layer = RandomSpanningTree(TreeFilter2D.norm2_distance, torch.exp)
        self.mst_layer = MinimumSpanningTree(TreeFilter2D.norm2_distance, torch.exp)
        self.tree_filter_layer  = TreeFilter2D(groups=num_groups)

        ''' Parameters init '''
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.conf_layer.weight, 0)
        nn.init.normal_(self.embed_layer.weight, std=0.01)
        nn.init.normal_(self.guide_layer.weight, std=0.01)
        nn.init.constant_(self.gamma, 0)
        nn.init.constant_(self.beta, 0)

    def split_groups(self, x):
        x = x.reshape(x.shape[0] * self.num_groups, -1, *x.shape[2:])
        return x

    def expand_groups(self, x):
        target_dim = max(self.num_groups // x.shape[1], 1)
        x = x.unsqueeze(2)
        x = x.expand(*x.shape[:2], target_dim, *x.shape[3:])
        x = x.reshape(x.shape[0], -1, *x.shape[3:])
        return x

    def forward(self, feature, guide):
        latent = feature

        ''' Compute embedding features '''
        embed = self.embed_layer(feature)

        ''' Spanning tree process '''
        guide = F.adaptive_avg_pool2d(guide, feature.shape[-2:])
        guide_embed = self.guide_layer(guide)
        if self.training:
            tree = self.rst_layer(guide_embed)
        else:
            tree = self.mst_layer(guide_embed)

        ''' Reshape beta '''
        beta = self.beta.reshape(1, -1, 1, 1)
        beta = beta.expand(embed.shape[0], self.num_groups, *embed.shape[2:])

        ''' Compute confidence '''
        conf = self.conf_layer(feature).sigmoid()
        conf = self.expand_groups(conf)
        conf_norm = self.tree_filter_layer(conf, embed, tree, guide_embed, beta)

        ''' Feature transform '''
        feature = (self.split_groups(feature) * self.split_groups(conf)).reshape_as(feature)
        feature = self.tree_filter_layer(feature, embed, tree, guide_embed, beta)
        feature_size = feature.size()
        feature = self.split_groups(feature) / (self.eps + self.split_groups(conf_norm))
        feature = feature.reshape(feature_size)

        ''' Projection '''
        feature = self.gamma * feature
        feature = feature + latent

        return feature
