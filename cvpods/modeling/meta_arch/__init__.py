# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.

# import all the meta_arch, so they will be registered

from .auto_assign import AutoAssign
from .borderdet import BorderDet
from .centernet import CenterNet
from .dynamic4seg import DynamicNet4Seg
from .efficientdet import EfficientDet
from .fcn import FCNHead
from .fcos import FCOS
from .free_anchor import FreeAnchor
from .panoptic_fpn import PanopticFPN
from .pointrend import CoarseMaskHead, PointRendROIHeads, PointRendSemSegHead, StandardPointHead
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .reppoints import RepPoints
from .retinanet import RetinaNet
from .semantic_seg import SemanticSegmentor, SemSegFPNHead
from .ssd import SSD
from .tensormask import TensorMask
from .yolov3 import YOLOv3
