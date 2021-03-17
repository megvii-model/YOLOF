# encoding: utf-8
# network file -> build backbone for Dynamic Network
# @author: yanwei.li
import numpy as np

import torch
import torch.nn as nn

from cvpods.layers import Conv2d, ShapeSpec, get_norm
from cvpods.modeling.backbone import Backbone
from cvpods.modeling.backbone.dynamic_arch import cal_op_flops
from cvpods.modeling.nn_utils import weight_init

from .dynamic_cell import Cell

__all__ = ["DynamicStem", "DynamicNetwork", "build_dynamic_backbone"]


class DynamicStem(nn.Module):
    def __init__(
        self, in_channels=3, mid_channels=64, out_channels=64,
        input_res=None, sept_stem=True, norm="BN", affine=True
    ):
        """
        Build basic STEM for Dynamic Network.
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()

        self.real_flops = 0.0
        # start with 3 stem layers down-sampling by 4.
        self.stem_1 = Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=2,
            bias=False, norm=get_norm(norm, mid_channels),
            activation=nn.ReLU()
        )
        self.real_flops += cal_op_flops.count_ConvBNReLU_flop(
            input_res[0], input_res[1], 3, mid_channels,
            [3, 3], stride=2, is_affine=affine
        )
        # stem 2
        input_res = input_res // 2
        if not sept_stem:
            self.stem_2 = Conv2d(
                mid_channels, mid_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
                norm=get_norm(norm, mid_channels),
                activation=nn.ReLU()
            )
            self.real_flops += cal_op_flops.count_ConvBNReLU_flop(
                input_res[0], input_res[1], mid_channels,
                mid_channels, [3, 3], is_affine=affine
            )
        else:
            self.stem_2 = nn.Sequential(
                Conv2d(
                    mid_channels, mid_channels, kernel_size=3, stride=1,
                    padding=1, groups=mid_channels, bias=False
                ),
                Conv2d(
                    mid_channels, mid_channels, kernel_size=1,
                    stride=1, padding=0, bias=False,
                    norm=get_norm(norm, mid_channels),
                    activation=nn.ReLU()
                )
            )
            self.real_flops += (
                cal_op_flops.count_Conv_flop(
                    input_res[0], input_res[1], mid_channels,
                    mid_channels, [3, 3], groups=mid_channels
                ) + cal_op_flops.count_ConvBNReLU_flop(
                    input_res[0], input_res[1], mid_channels,
                    mid_channels, [1, 1], is_affine=affine
                )
            )
        # stem 3
        if not sept_stem:
            self.stem_3 = Conv2d(
                mid_channels, out_channels, kernel_size=3,
                stride=2, padding=1, bias=False,
                norm=get_norm(norm, out_channels),
                activation=nn.ReLU()
            )
            self.real_flops += cal_op_flops.count_ConvBNReLU_flop(
                input_res[0], input_res[1], mid_channels, out_channels,
                [3, 3], stride=2, is_affine=affine
            )
        else:
            self.stem_3 = nn.Sequential(
                Conv2d(
                    mid_channels, mid_channels, kernel_size=3, stride=2,
                    padding=1, groups=mid_channels, bias=False
                ),
                Conv2d(
                    mid_channels, out_channels, kernel_size=1, padding=0,
                    bias=False, norm=get_norm(norm, out_channels),
                    activation=nn.ReLU()
                )
            )
            self.real_flops += (
                cal_op_flops.count_Conv_flop(
                    input_res[0], input_res[1], mid_channels,
                    mid_channels, [3, 3], stride=2, groups=mid_channels
                ) + cal_op_flops.count_ConvBNReLU_flop(
                    input_res[0] // 2, input_res[1] // 2, mid_channels,
                    out_channels, [1, 1], is_affine=affine
                )
            )
        self.out_res = input_res // 2
        self.out_cha = out_channels
        # using Kaiming init
        for layer in [self.stem_1, self.stem_2, self.stem_3]:
            for name, m in layer.named_modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in')
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stem_3(x)
        return x

    @property
    def out_channels(self):
        return self.out_cha

    @property
    def stride(self):
        return 4

    @property
    def out_resolution(self):
        return self.out_res

    @property
    def flops(self):
        return self.real_flops


class DynamicNetwork(Backbone):
    """
    This module implements Dynamic Routing Network.
    It creates dense connected network on top of some input feature maps.
    """
    def __init__(
        self, init_channel, input_shape, cell_num_list, layer_num,
        ext_layer=None, norm="", cal_flops=True, cell_type='',  # pylint: disable=W0613
        max_stride=32, sep_stem=True, using_gate=False,
        small_gate=False, gate_bias=1.5, drop_prob=0.0,
    ):
        super(DynamicNetwork, self).__init__()
        # set affine in BatchNorm
        if 'Sync' in norm:
            self.affine = True
        else:
            self.affine = False
        # set scheduled drop path
        self.drop_prob = drop_prob
        if self.drop_prob > 0.0001:
            self.drop_path = True
        else:
            self.drop_path = False
        self.cal_flops = cal_flops
        self._size_divisibility = max_stride
        input_res = np.array(input_shape[1:3])

        self.stem = DynamicStem(
            3, out_channels=init_channel, input_res=input_res,
            sept_stem=sep_stem, norm=norm, affine=self.affine
        )
        self.stem_flops = self.stem.flops
        self._out_feature_strides = {"stem": self.stem.stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}
        self._out_feature_resolution = {"stem": self.stem.out_resolution}
        assert self.stem.out_channels == init_channel
        self.all_cell_list = nn.ModuleList()
        self.all_cell_type_list = []
        self.cell_num_list = cell_num_list[:layer_num]
        self._out_features = []
        # using the initial layer
        input_res = input_res // self.stem.stride
        in_channel = out_channel = init_channel
        self.init_layer = Cell(
            C_in=in_channel, C_out=out_channel, norm=norm, allow_up=False,
            allow_down=True, input_size=input_res, cell_type=cell_type,
            cal_flops=False, using_gate=using_gate, small_gate=small_gate,
            gate_bias=gate_bias, affine=self.affine
        )

        # add cells in each layer
        for layer_index in range(len(self.cell_num_list)):
            layer_cell_list = nn.ModuleList()
            layer_cell_type = []
            for cell_index in range(self.cell_num_list[layer_index]):
                # channel multi, when stride:4 -> channel:C, stride:8 -> channel:2C ...
                channel_multi = pow(2, cell_index)
                in_channel_cell = in_channel * channel_multi
                # add res and dim switch to each cell
                allow_up = True
                allow_down = True
                # add res up and dim down by 2
                if cell_index == 0 or layer_index == layer_num - 1:
                    allow_up = False
                # dim down and resolution up by 2
                if cell_index == 3 or layer_index == layer_num - 1:
                    allow_down = False
                res_size = input_res // channel_multi
                layer_cell_list.append(
                    Cell(
                        C_in=in_channel_cell, C_out=in_channel_cell, norm=norm,
                        allow_up=allow_up, allow_down=allow_down,
                        input_size=res_size, cell_type=cell_type,
                        cal_flops=cal_flops, using_gate=using_gate,
                        small_gate=small_gate, gate_bias=gate_bias,
                        affine=self.affine
                    )
                )
                # allow dim change in each aggregation
                dim_up, dim_down, dim_keep = False, False, True
                # dim up and resolution down by 2
                if cell_index > 0:
                    dim_up = True
                # dim down and resolution up by 2
                if (cell_index < self.cell_num_list[layer_index] - 1) and layer_index > 2:
                    dim_down = True
                elif (cell_index < self.cell_num_list[layer_index] - 2) and layer_index <= 2:
                    dim_down = True
                # dim keep unchanged
                if layer_index <= 2 and cell_index == self.cell_num_list[layer_index] - 1:
                    dim_keep = False
                # allowed cell operations
                layer_cell_type.append([dim_up, dim_keep, dim_down])
                if layer_index == len(self.cell_num_list) - 1:
                    name = 'layer_' + str(cell_index)
                    self._out_feature_strides[name] = channel_multi * self.stem.stride
                    self._out_feature_channels[name] = in_channel_cell
                    self._out_feature_resolution[name] = res_size
                    self._out_features.append(name)
            self.all_cell_list.append(layer_cell_list)
            self.all_cell_type_list.append(layer_cell_type)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x, step_rate=0.0):
        h_l1 = self.stem(x)
        # the initial layer
        h_l1_list, h_beta_list, trans_flops, trans_flops_real = self.init_layer(h_l1=h_l1)
        prev_beta_list, prev_out_list = [h_beta_list], [h_l1_list]  # noqa: F841
        prev_trans_flops, prev_trans_flops_real = [trans_flops], [trans_flops_real]
        # build forward outputs
        cell_flops_list, cell_flops_real_list = [], []
        for layer_index in range(len(self.cell_num_list)):
            layer_input, layer_output = [], []
            layer_trans_flops, layer_trans_flops_real = [], []
            flops_in_expt_list, flops_in_real_list = [], []
            layer_rate = (layer_index + 1) / float(len(self.cell_num_list))
            # aggregate cell input
            for cell_index in range(len(self.all_cell_type_list[layer_index])):
                cell_input, trans_flops_input, trans_flops_real_input = [], [], []
                if self.all_cell_type_list[layer_index][cell_index][0]:
                    cell_input.append(prev_out_list[cell_index - 1][2][0])
                    trans_flops_input.append(prev_trans_flops[cell_index - 1][2][0])
                    trans_flops_real_input.append(prev_trans_flops_real[cell_index - 1][2][0])
                if self.all_cell_type_list[layer_index][cell_index][1]:
                    cell_input.append(prev_out_list[cell_index][1][0])
                    trans_flops_input.append(prev_trans_flops[cell_index][1][0])
                    trans_flops_real_input.append(prev_trans_flops_real[cell_index][1][0])
                if self.all_cell_type_list[layer_index][cell_index][2]:
                    cell_input.append(prev_out_list[cell_index + 1][0][0])
                    trans_flops_input.append(prev_trans_flops[cell_index + 1][0][0])
                    trans_flops_real_input.append(prev_trans_flops_real[cell_index + 1][0][0])

                h_l1 = sum(cell_input)
                # calculate input for gate
                layer_input.append(h_l1)
                # calculate FLOPs input
                flops_in_expt = sum(_flops for _flops in trans_flops_input)
                flop_in_real = sum(_flops for _flops in trans_flops_real_input)
                flops_in_expt_list.append(flops_in_expt)
                flops_in_real_list.append(flop_in_real)

            # calculate each cell
            for _cell_index in range(len(self.all_cell_type_list[layer_index])):
                if self.cal_flops:
                    cell_output, gate_weights_beta, cell_flops, \
                        cell_flops_real, trans_flops, trans_flops_real = \
                        self.all_cell_list[layer_index][_cell_index](
                            h_l1=layer_input[_cell_index],
                            flops_in_expt=flops_in_expt_list[_cell_index],
                            flops_in_real=flops_in_real_list[_cell_index],
                            is_drop_path=self.drop_path, drop_prob=self.drop_prob,
                            layer_rate=layer_rate, step_rate=step_rate
                        )
                    # calculate real flops
                    cell_flops_list.append(cell_flops)
                    cell_flops_real_list.append(cell_flops_real)
                else:
                    cell_output, gate_weights_beta, trans_flops, trans_flops_real = \
                        self.all_cell_list[layer_index][_cell_index](
                            h_l1=layer_input[_cell_index],
                            flops_in_expt=flops_in_expt_list[_cell_index],
                            flops_in_real=flops_in_real_list[_cell_index],
                            is_drop_path=self.drop_path, drop_prob=self.drop_prob,
                            layer_rate=layer_rate, step_rate=step_rate
                        )

                layer_output.append(cell_output)
                # update trans flops output
                layer_trans_flops.append(trans_flops)
                layer_trans_flops_real.append(trans_flops_real)
            # update layer output
            prev_out_list = layer_output
            prev_trans_flops = layer_trans_flops
            prev_trans_flops_real = layer_trans_flops_real

        final_out_list = [prev_out_list[_i][1][0] for _i in range(len(prev_out_list))]
        final_out_dict = dict(zip(self._out_features, final_out_list))
        if self.cal_flops:
            all_cell_flops = torch.mean(sum(cell_flops_list))
            all_flops_real = torch.mean(sum(cell_flops_real_list)) + self.stem_flops
        else:
            all_cell_flops, all_flops_real = None, None
        return final_out_dict, all_cell_flops, all_flops_real

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                height=self._out_feature_resolution[name][0],
                width=self._out_feature_resolution[name][0],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


def build_dynamic_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a Dynamic Backbone from config.
    Args:
        cfg: a config dict
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
    backbone = DynamicNetwork(
        init_channel=cfg.MODEL.BACKBONE.INIT_CHANNEL,
        input_shape=input_shape,
        cell_num_list=cfg.MODEL.BACKBONE.CELL_NUM_LIST,
        layer_num=cfg.MODEL.BACKBONE.LAYER_NUM,
        norm=cfg.MODEL.BACKBONE.NORM,
        cal_flops=cfg.MODEL.CAL_FLOPS,
        cell_type=cfg.MODEL.BACKBONE.CELL_TYPE,
        max_stride=cfg.MODEL.BACKBONE.MAX_STRIDE,
        sep_stem=cfg.MODEL.BACKBONE.SEPT_STEM,
        using_gate=cfg.MODEL.GATE.GATE_ON,
        small_gate=cfg.MODEL.GATE.SMALL_GATE,
        gate_bias=cfg.MODEL.GATE.GATE_INIT_BIAS,
        drop_prob=cfg.MODEL.BACKBONE.DROP_PROB
    )

    return backbone
