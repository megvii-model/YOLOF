import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.layers import Conv2dSamePadding as Conv2d
from cvpods.layers import MaxPool2dSamePadding as MaxPool2d
from cvpods.layers import MemoryEfficientSwish, SeparableConvBlock, Swish, get_norm
from cvpods.modeling.backbone import Backbone

from .efficientnet import build_efficientnet_backbone
from .fpn import _assert_strides_are_log2_contiguous


class BiFPNLayer(nn.Module):
    """
    This module implements one layer of BiFPN, and BiFPN can be obtained
    by stacking this module multiple times.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, input_size, in_channels_list, out_channels,
                 fuse_type="fast", norm="BN", memory_efficient=True):
        """
        input_size (int): the input image size.
        in_channels_list (list): the number of input tensor channels per level.
        out_channels (int): the number of output tensor channels.
        fuse_type (str): now only support three weighted fusion approaches:

            * fast:    Output = sum(Input_i * w_i / sum(w_j))
            * sotfmax: Output = sum(Input_i * e ^ w_i / sum(e ^ w_j))
            * sum:     Output = sum(Input_i) / len(Input_i)

        norm (str): the normalization to use.
        memory_efficient (bool): use `MemoryEfficientSwish` or `Swish` as activation function.
        """
        super(BiFPNLayer, self).__init__()
        assert fuse_type in ("fast", "softmax", "sum"), f"Unknown fuse method: {fuse_type}." \
            " Please select in [fast, sotfmax, sum]."

        self.input_size = input_size
        self.in_channels_list = in_channels_list
        self.fuse_type = fuse_type
        self.levels = len(in_channels_list)
        self.nodes_input_offsets = [
            [3, 4],
            [2, 5],
            [1, 6],
            [0, 7],
            [1, 7, 8],
            [2, 6, 9],
            [3, 5, 10],
            [4, 11],
        ]
        self.nodes_strides = [
            2 ** x
            for x in [6, 5, 4, 3, 4, 5, 6, 7]
        ]

        # Change input feature map to have target number of channels.
        self.resample_convs = nn.ModuleList()
        for node_i_input_offsets in self.nodes_input_offsets:
            resample_convs_i = nn.ModuleList()
            for input_offset in node_i_input_offsets:
                if self.in_channels_list[input_offset] != out_channels:
                    resample_conv = Conv2d(
                        self.in_channels_list[input_offset],
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        norm=get_norm(norm, out_channels),
                        activation=None,
                    )
                else:
                    resample_conv = nn.Identity()
                self.in_channels_list.append(out_channels)
                resample_convs_i.append(resample_conv)
            self.resample_convs.append(resample_convs_i)

        # fpn combine weights
        self.edge_weights = nn.ParameterList()
        for node_i_input_offsets in self.nodes_input_offsets:
            # combine weight
            if fuse_type == "fast" or fuse_type == "softmax":
                weights_i = nn.Parameter(
                    torch.ones(len(node_i_input_offsets), dtype=torch.float32),
                    requires_grad=True,
                )
            elif fuse_type == "sum":
                weights_i = nn.Parameter(
                    torch.ones(len(node_i_input_offsets), dtype=torch.float32),
                    requires_grad=False,
                )
            else:
                raise ValueError("Unknown fuse method: {}".format(self.fuse_type))
            self.edge_weights.append(weights_i)

        # Convs for combine edge features
        self.combine_convs = nn.ModuleList()
        for node_i_input_offsets in self.nodes_input_offsets:
            combine_conv = SeparableConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding="SAME",
                norm=get_norm(norm, out_channels),
                activation=None,
            )
            self.combine_convs.append(combine_conv)

        self.act = MemoryEfficientSwish() if memory_efficient else Swish()
        self.down_sampling = MaxPool2d(kernel_size=3, stride=2, padding="SAME")
        self.up_sampling = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # Build top-down and bottom-up path
        self.nodes_features = inputs
        for node_idx, (node_i_input_offsets, node_i_stride) in enumerate(
                zip(self.nodes_input_offsets, self.nodes_strides)):
            # edge weights
            if self.fuse_type == "fast":
                weights_i = F.relu(self.edge_weights[node_idx])
            elif self.fuse_type == "softmax":
                weights_i = self.edge_weights[node_idx].softmax(dim=0)
            elif self.fuse_type == "sum":
                weights_i = self.edge_weights[node_idx]

            target_width = self.input_size / node_i_stride
            edge_features = []
            for offset_idx, offset in enumerate(node_i_input_offsets):
                edge_feature = self.nodes_features[offset]
                resample_conv = self.resample_convs[node_idx][offset_idx]
                # 1x1 conv for change feature map channels if necessary
                edge_feature = resample_conv(edge_feature)
                width = edge_feature.size(-1)
                if width > target_width:
                    # Downsampling for change feature map size
                    assert width / target_width == 2.0
                    edge_feature = self.down_sampling(edge_feature)
                elif width < target_width:
                    # Upsampling for change feature map size
                    assert target_width / width == 2.0
                    edge_feature = self.up_sampling(edge_feature)
                edge_feature = edge_feature * (weights_i[offset_idx] / (weights_i.sum() + 1e-4))
                edge_features.append(edge_feature)
            node_i_feature = sum(edge_features)
            node_i_feature = self.act(node_i_feature)
            node_i_feature = self.combine_convs[node_idx](node_i_feature)
            self.nodes_features.append(node_i_feature)

        # The number of node in one bifpn layer is 13
        assert len(self.nodes_features) == 13
        # The bifpn layer output is the last 5 nodes
        return self.nodes_features[-5:]


class BiFPN(Backbone):
    """
    This module implements the BIFPN module in EfficientDet.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, input_size, bottom_up, in_features, out_channels, num_bifpn_layers,
                 fuse_type="weighted_sum", top_block=None, norm="BN", bn_momentum=0.01, bn_eps=1e-3,
                 memory_efficient=True):
        """
        input_size (int): the input image size.
        bottom_up (Backbone): module representing the bottom up subnetwork.
            Must be a subclass of :class:`Backbone`. The multi-scale feature
            maps generated by the bottom up network, and listed in `in_features`,
            are used to generate FPN levels.
        in_features (list[str]): names of the input feature maps coming
            from the backbone to which FPN is attached. For example, if the
            backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
            of these may be used; order must be from high to low resolution.
        out_channels (int): the number of channels in the output feature maps.
        num_bifpn_layers (str): the number of bifpn layer.
        fuse_type (str): weighted feature fuse type. see: `BiFPNLayer`
        top_block (nn.Module or None): if provided, an extra operation will
            be performed on the output of the last (smallest resolution)
            FPN output, and the result will extend the result list. The top_block
            further downsamples the feature map. It must have an attribute
            "num_levels", meaning the number of extra FPN levels added by
            this block, and "in_feature", which is a string representing
            its input feature (e.g., p5).
        norm (str): the normalization to use.
        bn_momentum (float): the `momentum` parameter of the norm module.
        bn_eps (float): the `eps` parameter of the norm module.
        """
        super(BiFPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        self.bottom_up = bottom_up
        self.top_block = top_block
        self.in_features = in_features
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps

        # Feature map strides and channels from the bottom up network
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}

        # top block output feature maps.
        if self.top_block is not None:
            s = int(math.log2(in_strides[-1]))
            for i in range(self.top_block.num_levels):
                self._out_feature_strides[f"p{s + i + 1}"] = 2 ** (s + i + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}

        # build bifpn layers
        self.bifpn_layers = nn.ModuleList()
        for idx in range(num_bifpn_layers):
            if idx == 0:
                bifpn_layer_in_channels = in_channels + [out_channels] * self.top_block.num_levels
            else:
                bifpn_layer_in_channels = [out_channels] * len(self._out_features)
            bifpn_layer = BiFPNLayer(input_size, bifpn_layer_in_channels,
                                     out_channels, fuse_type, norm, memory_efficient)
            self.bifpn_layers.append(bifpn_layer)

        self._size_divisibility = in_strides[-1]
        self._init_weights()

    def _init_weights(self):
        """
        Weight initialization as per Tensorflow official implementations.
        See: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/init_ops.py
             #L437
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stddev = math.sqrt(1. / max(1., fan_in))
                m.weight.data.normal_(0, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if self.bn_momentum is not None and self.bn_eps is not None:
                    m.momentum = self.bn_momentum
                    m.eps = self.bn_eps
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        results = [bottom_up_features[f] for f in self.in_features]

        # top block
        if self.top_block is not None:
            top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            results.extend(self.top_block(top_block_in_feature))

        # build top-down and bottom-up path with stack
        for bifpn_layer in self.bifpn_layers:
            results = bifpn_layer(results)
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))


class BiFPNP6P7(nn.Module):
    """
    This module is used in BiFPN to generate extra layers,
    P6 and P7 from EfficientNet "stage8" feature.
    """

    def __init__(self, in_channels, out_channels, norm="BN"):
        """
        Args:
            in_channels (int): the number of input tensor channels.
            out_channels (int): the number of output tensor channels.
            norm (str): the normalization to use.
        """
        super().__init__()
        self.num_levels = 2
        self.in_feature = "stage8"
        self.p6_conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=get_norm(norm, out_channels),
            activation=None
        )
        self.down_sampling = MaxPool2d(kernel_size=3, stride=2, padding="SAME")

    def forward(self, x):
        x = self.p6_conv(x)
        p6 = self.down_sampling(x)
        p7 = self.down_sampling(p6)
        return [p6, p7]


def build_efficientnet_bifpn_backbone(cfg, input_shape):
    """
    Args:
        cfg: a cvpods `Config` instance.

    Returns:
        bifpn (Backbone): backbone module, must be a subclass of
            :class:`Backbone`.
    """
    in_features = cfg.MODEL.BIFPN.IN_FEATURES
    norm = cfg.MODEL.BIFPN.NORM
    bn_momentum = cfg.MODEL.BIFPN.BN_MOMENTUM
    bn_eps = cfg.MODEL.BIFPN.BN_EPS
    memory_efficient = cfg.MODEL.BIFPN.MEMORY_EFFICIENT_SWISH
    input_size = cfg.MODEL.BIFPN.INPUT_SIZE
    out_channels = cfg.MODEL.BIFPN.OUT_CHANNELS
    bifpn_layers = cfg.MODEL.BIFPN.NUM_LAYERS
    fuse_type = cfg.MODEL.BIFPN.FUSE_TYPE

    # when norm is "" or None, set norm = "BN"
    if not norm:
        norm = "BN"

    bottom_up = build_efficientnet_backbone(cfg, input_shape)
    top_block = BiFPNP6P7(
        bottom_up.output_shape()[in_features[-1]].channels,
        out_channels, norm)
    bifpn = BiFPN(input_size, bottom_up, in_features, out_channels,
                  bifpn_layers, fuse_type, top_block, norm, bn_momentum,
                  bn_eps, memory_efficient)
    return bifpn
