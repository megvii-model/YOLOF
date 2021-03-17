# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.

from .backbone import Backbone
from .bifpn import BiFPN, build_efficientnet_bifpn_backbone
from .darknet import Darknet, build_darknet_backbone
from .dynamic_arch import DynamicNetwork, build_dynamic_backbone
from .efficientnet import EfficientNet, build_efficientnet_backbone
from .fpn import (
    FPN,
    build_retinanet_mobilenetv2_fpn_p5_backbone,
    build_retinanet_resnet_fpn_p5_backbone
)
from .mobilenet import InvertedResBlock, MobileNetV2, MobileStem, build_mobilenetv2_backbone
from .resnet import ResNet, ResNetBlockBase, build_resnet_backbone, make_stage
# TODO can expose more resnet blocks after careful consideration
from .shufflenet import ShuffleNetV2, ShuffleV2Block, build_shufflenetv2_backbone
from .snet import SNet, build_snet_backbone
from .transformer import Transformer
