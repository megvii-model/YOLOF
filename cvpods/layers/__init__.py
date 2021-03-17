# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection, Inc. and its affiliates. All Rights Reserved
from .activation_funcs import MemoryEfficientSwish, Swish
from .batch_norm import FrozenBatchNorm2d, NaiveSyncBatchNorm, get_activation, get_norm
from .deform_conv import DeformConv, ModulatedDeformConv
from .deform_conv_with_off import DeformConvWithOff, ModulatedDeformConvWithOff
from .larc import LARC
from .mask_ops import paste_masks_in_image
from .nms import (
    batched_nms,
    batched_nms_rotated,
    batched_softnms,
    batched_softnms_rotated,
    cluster_nms,
    generalized_batched_nms,
    matrix_nms,
    ml_nms,
    nms,
    nms_rotated,
    softnms,
    softnms_rotated
)
from .position_encoding import position_encoding_dict
from .roi_align import ROIAlign, roi_align
from .roi_align_rotated import ROIAlignRotated, roi_align_rotated
from .shape_spec import ShapeSpec
from .swap_align2nat import SwapAlign2Nat, swap_align2nat
from .tree_filter_v2 import TreeFilterV2
from .wrappers import (
    BatchNorm2d,
    Conv2d,
    Conv2dSamePadding,
    ConvTranspose2d,
    MaxPool2dSamePadding,
    SeparableConvBlock,
    cat,
    interpolate
)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
