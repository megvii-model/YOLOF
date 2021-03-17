#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from cvpods import _C

# pylint: disable=W0613


class _BFS(Function):
    @staticmethod
    def forward(ctx, edge_index, max_adj_per_vertex):
        sorted_index, sorted_parent, sorted_child =\
            _C.bfs_forward(edge_index, max_adj_per_vertex)
        return sorted_index, sorted_parent, sorted_child


class _MST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.mst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None


class _RST(Function):
    @staticmethod
    def forward(ctx, edge_index, edge_weight, vertex_index):
        edge_out = _C.rst_forward(edge_index, edge_weight, vertex_index)
        return edge_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return None, None, None


class _Refine(Function):
    @staticmethod
    def forward(ctx, feature_in, edge_weight, self_weight,
                sorted_index, sorted_parent, sorted_child):
        feature_out, feature_aggr, feature_aggr_up, =\
            _C.tree_filter_refine_forward(
                feature_in, edge_weight, self_weight,
                sorted_index, sorted_parent, sorted_child
            )

        ctx.save_for_backward(
            feature_in, edge_weight, self_weight, sorted_index,
            sorted_parent, sorted_child, feature_aggr, feature_aggr_up
        )
        return feature_out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_in, edge_weight, self_weight, sorted_index, sorted_parent,\
            sorted_child, feature_aggr, feature_aggr_up = ctx.saved_tensors

        grad_feature = _C.tree_filter_refine_backward_feature(
            feature_in, edge_weight, self_weight, sorted_index,
            sorted_parent, sorted_child, feature_aggr, feature_aggr_up,
            grad_output
        )
        grad_edge_weight = _C.tree_filter_refine_backward_edge_weight(
            feature_in, edge_weight, self_weight, sorted_index, sorted_parent,
            sorted_child, feature_aggr, feature_aggr_up, grad_output
        )
        grad_self_weight = _C.tree_filter_refine_backward_self_weight(
            feature_in, edge_weight, self_weight, sorted_index, sorted_parent,
            sorted_child, feature_aggr, feature_aggr_up, grad_output
        )

        return grad_feature, grad_edge_weight, grad_self_weight, None, None, None


bfs = _BFS.apply
mst = _MST.apply
rst = _RST.apply
refine = _Refine.apply


class MinimumSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(MinimumSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2),
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(weight)
        return weight

    def forward(self, guide_in):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            weight = self._build_feature_weight(guide_in)
            tree = mst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree


class RandomSpanningTree(nn.Module):
    def __init__(self, distance_func, mapping_func=None):
        super(RandomSpanningTree, self).__init__()
        self.distance_func = distance_func
        self.mapping_func = mapping_func

    @staticmethod
    def _build_matrix_index(fm):
        batch, height, width = (fm.shape[0], *fm.shape[2:])
        row = torch.arange(width, dtype=torch.int32, device=fm.device).unsqueeze(0)
        col = torch.arange(height, dtype=torch.int32, device=fm.device).unsqueeze(1)
        raw_index = row + col * width
        row_index = torch.stack([raw_index[:-1, :], raw_index[1:, :]], 2)
        col_index = torch.stack([raw_index[:, :-1], raw_index[:, 1:]], 2)
        index = torch.cat([row_index.reshape(1, -1, 2),
                           col_index.reshape(1, -1, 2)], 1)
        index = index.expand(batch, -1, -1)
        return index

    def _build_feature_weight(self, fm):
        batch = fm.shape[0]
        weight_row = self.distance_func(fm[:, :, :-1, :], fm[:, :, 1:, :])
        weight_col = self.distance_func(fm[:, :, :, :-1], fm[:, :, :, 1:])
        weight_row = weight_row.reshape([batch, -1])
        weight_col = weight_col.reshape([batch, -1])
        weight = torch.cat([weight_row, weight_col], dim=1)
        if self.mapping_func is not None:
            weight = self.mapping_func(-weight)
        return weight

    def forward(self, guide_in):
        with torch.no_grad():
            index = self._build_matrix_index(guide_in)
            weight = self._build_feature_weight(guide_in)
            tree = rst(index, weight, guide_in.shape[2] * guide_in.shape[3])
        return tree


class TreeFilter2D(nn.Module):
    def __init__(self, groups=1, distance_func=None,
                 mapping_func=torch.exp):
        super(TreeFilter2D, self).__init__()
        self.groups = groups
        self.mapping_func = mapping_func
        if distance_func is None:
            self.distance_func = self.norm2_distance
        else:
            self.distance_func = distance_func

    @staticmethod
    def norm2_distance(fm_ref, fm_tar):
        diff = fm_ref - fm_tar
        weight = (diff * diff).sum(dim=1)
        return weight

    @staticmethod
    def batch_index_opr(data, index):
        with torch.no_grad():
            channel = data.shape[1]
            index = index.unsqueeze(1).expand(-1, channel, -1).long()
        data = torch.gather(data, 2, index)
        return data

    def build_edge_weight(self, fm, sorted_index, sorted_parent):
        batch   = fm.shape[0]
        channel = fm.shape[1]
        vertex  = fm.shape[2] * fm.shape[3]

        fm = fm.reshape([batch, channel, -1])
        fm_source = self.batch_index_opr(fm, sorted_index)
        fm_target = self.batch_index_opr(fm_source, sorted_parent)
        fm_source = fm_source.reshape([-1, channel // self.groups, vertex])
        fm_target = fm_target.reshape([-1, channel // self.groups, vertex])

        edge_weight = self.distance_func(fm_source, fm_target)
        edge_weight = self.mapping_func(-edge_weight)
        return edge_weight

    def build_self_weight(self, fm, sorted_index):
        vertex = fm.shape[2] * fm.shape[3]

        fm = fm.reshape(-1, fm.shape[1] // self.groups, vertex)
        self_dist = self.distance_func(fm, 0)
        self_weight = self.mapping_func(-self_dist)
        att_weight = self_weight.reshape(-1, self.groups, vertex)
        att_weight = self.batch_index_opr(att_weight, sorted_index)
        att_weight = att_weight.reshape(-1, vertex)
        return self_weight, att_weight

    def split_group(self, feature_in, *tree_orders):
        feature_in = feature_in.reshape(
            feature_in.shape[0] * self.groups,
            feature_in.shape[1] // self.groups,
            -1
        )
        returns = [feature_in.contiguous()]
        for order in tree_orders:
            order = order.unsqueeze(1).expand(order.shape[0], self.groups, *order.shape[1:])
            order = order.reshape(-1, *order.shape[2:])
            returns.append(order.contiguous())
        return tuple(returns)

    def forward(self, feature_in, embed_in, tree, guide_in=None, self_dist_in=None):
        ori_shape = feature_in.shape
        sorted_index, sorted_parent, sorted_child = bfs(tree, 4)
        edge_weight = self.build_edge_weight(embed_in, sorted_index, sorted_parent)
        if self_dist_in is None:
            self_weight = torch.ones_like(edge_weight)
        else:
            self_weight, att_weight = self.build_self_weight(self_dist_in, sorted_index)
            edge_weight = edge_weight * att_weight

        if guide_in is not None:
            guide_weight = self.build_edge_weight(guide_in, sorted_index, sorted_parent)
            edge_weight = edge_weight * guide_weight

        feature_in, sorted_index, sorted_parent, sorted_child = \
            self.split_group(feature_in, sorted_index, sorted_parent, sorted_child)
        feature_out = refine(feature_in, edge_weight, self_weight, sorted_index,
                             sorted_parent, sorted_child)
        feature_out = feature_out.reshape(ori_shape)
        return feature_out
