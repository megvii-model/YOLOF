# encoding: utf-8
# network file -> build Cell for Dynamic Backbone
# @author: yanwei.li
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.layers import Conv2d, get_norm
from cvpods.modeling.backbone.dynamic_arch import cal_op_flops
from cvpods.modeling.nn_utils import weight_init

from .op_with_flops import OPS, Identity

__all__ = ["Mixed_OP", "Cell"]


# soft gate for path choice
def soft_gate(x, x_t=None, momentum=0.1, is_update=False):
    if is_update:
        # using momentum for weight update
        y = (1 - momentum) * x.data + momentum * x_t
        tanh_value = torch.tanh(y)
        return F.relu(tanh_value), y.data
    else:
        tanh_value = torch.tanh(x)
        return F.relu(tanh_value)


# Scheduled Drop Path
def drop_path(x, drop_prob, layer_rate, step_rate):
    """
    :param x: input feature
    :param drop_prob: drop path prob
    :param layer_rate: current_layer/total_layer
    :param step_rate: current_step/total_step
    :return: output feature
    """
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        keep_prob = 1. - layer_rate * (1. - keep_prob)
        keep_prob = 1. - step_rate * (1. - keep_prob)
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Mixed_OP(nn.Module):
    """
    Sum up operations according to their weights.
    """
    def __init__(
        self, inplanes, outplanes, stride, cell_type,
        norm='', affine=True, input_size=None
    ):
        super(Mixed_OP, self).__init__()
        self._ops = nn.ModuleList()
        self.op_flops = []
        for key in cell_type:
            op = OPS[key](
                inplanes, outplanes, stride, norm_layer=norm,
                affine=affine, input_size=input_size
            )
            self._ops.append(op)
            self.op_flops.append(op.flops)
        self.real_flops = sum(op_flop for op_flop in self.op_flops)

    def forward(self, x, is_drop_path=False, drop_prob=0.0, layer_rate=0.0, step_rate=0.0):
        if is_drop_path:
            y = []
            for op in self._ops:
                if not isinstance(op, Identity):
                    y.append(drop_path(op(x), drop_prob, layer_rate, step_rate))
                else:
                    y.append(op(x))
            return sum(y)
        else:
            # using sum up rather than random choose one branch.
            return sum(op(x) for op in self._ops)

    @property
    def flops(self):
        return self.real_flops.squeeze()


class Cell(nn.Module):
    def __init__(  # noqa:C901
        self, C_in, C_out, norm, allow_up, allow_down, input_size,
        cell_type, cal_flops=True, using_gate=False,
        small_gate=False, gate_bias=1.5, affine=True
    ):
        super(Cell, self).__init__()
        self.channel_in = C_in
        self.channel_out = C_out
        self.allow_up = allow_up
        self.allow_down = allow_down
        self.cal_flops = cal_flops
        self.using_gate = using_gate
        self.small_gate = small_gate

        self.cell_ops = Mixed_OP(
            inplanes=self.channel_in, outplanes=self.channel_out,
            stride=1, cell_type=cell_type, norm=norm,
            affine=affine, input_size=input_size
        )
        self.cell_flops = self.cell_ops.flops
        # resolution keep
        self.res_keep = nn.ReLU()
        self.res_keep_flops = cal_op_flops.count_ReLU_flop(
            input_size[0], input_size[1], self.channel_out
        )
        # resolution up and dim down
        if self.allow_up:
            self.res_up = nn.Sequential(
                nn.ReLU(),
                Conv2d(
                    self.channel_out, self.channel_out // 2, kernel_size=1,
                    stride=1, padding=0, bias=False,
                    norm=get_norm(norm, self.channel_out // 2),
                    activation=nn.ReLU()
                )
            )
            # calculate Flops
            self.res_up_flops = cal_op_flops.count_ReLU_flop(
                input_size[0], input_size[1], self.channel_out
            ) + cal_op_flops.count_ConvBNReLU_flop(
                input_size[0], input_size[1], self.channel_out,
                self.channel_out // 2, [1, 1], is_affine=affine
            )
            # using Kaiming init
            for m in self.res_up.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in')
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        # resolution down and dim up
        if self.allow_down:
            self.res_down = nn.Sequential(
                nn.ReLU(),
                Conv2d(
                    self.channel_out, 2 * self.channel_out,
                    kernel_size=1, stride=2, padding=0, bias=False,
                    norm=get_norm(norm, 2 * self.channel_out),
                    activation=nn.ReLU()
                )
            )
            # calculate Flops
            self.res_down_flops = cal_op_flops.count_ReLU_flop(
                input_size[0], input_size[1], self.channel_out
            ) + cal_op_flops.count_ConvBNReLU_flop(
                input_size[0], input_size[1], self.channel_out,
                2 * self.channel_out, [1, 1], stride=2, is_affine=affine
            )
            # using Kaiming init
            for m in self.res_down.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in')
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        if self.allow_up and self.allow_down:
            self.gate_num = 3
        elif self.allow_up or self.allow_down:
            self.gate_num = 2
        else:
            self.gate_num = 1
        if self.using_gate:
            self.gate_conv_beta = nn.Sequential(
                Conv2d(
                    self.channel_in, self.channel_in // 2, kernel_size=1,
                    stride=1, padding=0, bias=False,
                    norm=get_norm(norm, self.channel_in // 2),
                    activation=nn.ReLU()
                ),
                nn.AdaptiveAvgPool2d((1, 1)),
                Conv2d(
                    self.channel_in // 2, self.gate_num, kernel_size=1,
                    stride=1, padding=0, bias=True
                )
            )
            if self.small_gate:
                input_size = input_size // 4
            self.gate_flops = cal_op_flops.count_ConvBNReLU_flop(
                input_size[0], input_size[1], self.channel_in,
                self.channel_in // 2, [1, 1], is_affine=affine
            ) + cal_op_flops.count_Pool2d_flop(
                input_size[0], input_size[1], self.channel_in // 2, [1, 1], 1
            ) + cal_op_flops.count_Conv_flop(
                1, 1, self.channel_in // 2, self.gate_num, [1, 1]
            )
            # using Kaiming init and predefined bias for gate
            for m in self.gate_conv_beta.modules():
                if isinstance(m, nn.Conv2d):
                    weight_init.kaiming_init(m, mode='fan_in', bias=gate_bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            self.register_buffer(
                'gate_weights_beta', torch.ones(1, self.gate_num, 1, 1).cuda()
            )
            self.gate_flops = 0.0

    def forward(
        self, h_l1, flops_in_expt=None, flops_in_real=None,
        is_drop_path=False, drop_prob=0.0,
        layer_rate=0.0, step_rate=0.0
    ):
        """
        :param h_l1: # the former hidden layer output
        :return: current hidden cell result h_l
        """
        drop_cell = False
        # drop the cell if input type is float
        if not isinstance(h_l1, float):
            # calculate soft conditional gate
            if self.using_gate:
                if self.small_gate:
                    h_l1_gate = F.interpolate(
                        input=h_l1, scale_factor=0.25,
                        mode='bilinear', align_corners=False
                    )
                else:
                    h_l1_gate = h_l1
                gate_feat_beta = self.gate_conv_beta(h_l1_gate)
                gate_weights_beta = soft_gate(gate_feat_beta)
            else:
                gate_weights_beta = self.gate_weights_beta
        else:
            drop_cell = True
        # use for inference
        if not self.training:
            if not drop_cell:
                drop_cell = gate_weights_beta.sum() < 0.0001
            if drop_cell:
                result_list = [[0.0], [h_l1], [0.0]]
                weights_list_beta = [[0.0], [0.0], [0.0]]
                trans_flops_expt = [[0.0], [0.0], [0.0]]
                trans_flops_real = [[0.0], [0.0], [0.0]]
                if self.cal_flops:
                    h_l_flops = flops_in_expt
                    h_l_flops_real = flops_in_real + self.gate_flops
                    return (
                        result_list, weights_list_beta, h_l_flops,
                        h_l_flops_real, trans_flops_expt, trans_flops_real
                    )
                else:
                    return (
                        result_list, weights_list_beta,
                        trans_flops_expt, trans_flops_real
                    )

        h_l = self.cell_ops(h_l1, is_drop_path, drop_prob, layer_rate, step_rate)

        # resolution and dimension change
        # resolution: [up, keep, down]
        h_l_keep = self.res_keep(h_l)
        gate_weights_beta_keep = gate_weights_beta[:, 0].unsqueeze(-1)
        # using residual connection if drop cell
        gate_mask = (gate_weights_beta.sum(dim=1, keepdim=True) < 0.0001).float()
        result_list = [[], [gate_mask * h_l1 + gate_weights_beta_keep * h_l_keep], []]
        weights_list_beta = [[], [gate_mask * 1.0 + gate_weights_beta_keep], []]
        # calculate flops for keep res
        gate_mask_keep = (gate_weights_beta_keep > 0.0001).float()
        trans_flops_real = [[], [gate_mask_keep * self.res_keep_flops], []]
        # calculate trans flops
        trans_flops_expt = [[], [self.res_keep_flops * gate_weights_beta_keep], []]

        if self.allow_up:
            h_l_up = self.res_up(h_l)
            h_l_up = F.interpolate(
                input=h_l_up, scale_factor=2, mode='bilinear', align_corners=False
            )
            gate_weights_beta_up = gate_weights_beta[:, 1].unsqueeze(-1)
            result_list[0].append(h_l_up * gate_weights_beta_up)
            weights_list_beta[0].append(gate_weights_beta_up)
            trans_flops_expt[0].append(self.res_up_flops * gate_weights_beta_up)
            # calculate flops for up res
            gate_mask_up = (gate_weights_beta_up > 0.0001).float()
            trans_flops_real[0].append(gate_mask_up * self.res_up_flops)

        if self.allow_down:
            h_l_down = self.res_down(h_l)
            gate_weights_beta_down = gate_weights_beta[:, -1].unsqueeze(-1)
            result_list[2].append(h_l_down * gate_weights_beta_down)
            weights_list_beta[2].append(gate_weights_beta_down)
            trans_flops_expt[2].append(self.res_down_flops * gate_weights_beta_down)
            # calculate flops for down res
            gate_mask_down = (gate_weights_beta_down > 0.0001).float()
            trans_flops_real[2].append(gate_mask_down * self.res_down_flops)

        if self.cal_flops:
            cell_flops = gate_weights_beta.max(dim=1, keepdim=True)[0] * self.cell_flops
            cell_flops_real = (
                gate_weights_beta.sum(dim=1, keepdim=True) > 0.0001
            ).float() * self.cell_flops
            h_l_flops = cell_flops + flops_in_expt
            h_l_flops_real = cell_flops_real + flops_in_real + self.gate_flops
            return (
                result_list, weights_list_beta, h_l_flops,
                h_l_flops_real, trans_flops_expt, trans_flops_real
            )
        else:
            return result_list, weights_list_beta, trans_flops_expt, trans_flops_real
