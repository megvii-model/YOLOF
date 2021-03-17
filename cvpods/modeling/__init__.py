# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

from cvpods.layers import ShapeSpec

# from .anchor_generator import build_anchor_generator
from .backbone import FPN, Backbone, ResNet, ResNetBlockBase, build_resnet_backbone, make_stage
from .matcher import Matcher
from .meta_arch import GeneralizedRCNN, PanopticFPN, ProposalNetwork, RetinaNet, SemanticSegmentor
from .postprocessing import detector_postprocess
from .roi_heads import ROIHeads, StandardROIHeads
from .test_time_augmentation import DatasetMapperTTA, GeneralizedRCNNWithTTA, TTAWarper

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

assert (
    torch.Tensor([1]) == torch.Tensor([2])
).dtype == torch.bool, ("Your Pytorch is too old. "
                        "Please update to contain https://github.com/pytorch/pytorch/pull/21113")
