# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from enum import Enum
from typing import Callable, Iterable, Type, Union

import torch

from cvpods.layers import LARC

from .optimizer_builder import OPTIMIZER_BUILDER
from .scheduler_builder import SCHEDULER_BUILDER

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer_type: Type[torch.optim.Optimizer], gradient_clipper: _GradientClipper
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(cfg, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.
    Args:
        cfg: config dict
            configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance
    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer


def maybe_use_lars_optimizer(cfg, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
    """
    warp optimizer with LARS, see :clsss:`cvpods.solver.lars.LARS` for more information.

    Args:
        cfg(BaseConfig):
        optimizer(Optimizer): optimizer for warp

    Return:
        optimizer(Optimizer): optimizer with LARS warped
    """

    if hasattr(cfg.SOLVER.OPTIMIZER, "LARC") and cfg.SOLVER.OPTIMIZER.LARC.ENABLED:
        eps = cfg.SOLVER.OPTIMIZER.LARC.EPS
        trust_coef = cfg.SOLVER.OPTIMIZER.LARC.TRUST_COEF
        clip = cfg.SOLVER.OPTIMIZER.LARC.CLIP
        optimizer = LARC(optimizer, eps, trust_coef, clip)

    return optimizer


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer with clip and LARS wraper from config.
    """
    def map_name(name):
        map_dict = {
            "SGD": "SGDBuilder",
            "D2SGD": "D2SGDBuilder",  # Detectron2's SGD
            "Adam": "AdamBuilder",
            "AdamW": "AdamWBuilder",
            "SGD_GATE_LR_MULTI": "SGDGateLRBuilder",
        }
        if name in map_dict:
            name = map_dict[name]
        return name

    NAME = map_name(cfg.SOLVER.OPTIMIZER.NAME)
    assert NAME in OPTIMIZER_BUILDER, "Please registry your Optimizer Builder first."

    optimizer = OPTIMIZER_BUILDER.get(NAME).build(model, cfg)

    # warp optimizer
    optimizer = maybe_use_lars_optimizer(cfg, optimizer)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def build_lr_scheduler(
    cfg, optimizer: torch.optim.Optimizer, **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    def map_name(name):
        map_dict = {
            "WarmupMultiStepLR": "WarmupMultiStepLRBuilder",
            "WarmupCosineLR": "WarmupCosineLRBuilder",
            "PolyLR": "PolyLRBuilder",
            "LambdaLR": "LambdaLRBuilder",
            "OneCycleLR": "OneCycleLRBuilder",
        }
        if name in map_dict:
            name = map_dict[name]
        return name

    name = map_name(cfg.SOLVER.LR_SCHEDULER.NAME)
    assert name in SCHEDULER_BUILDER, "Please registry {} in SCHEDULER_BUILDER".format(name)

    scheduler = SCHEDULER_BUILDER.get(name).build(optimizer, cfg, **kwargs)
    return scheduler
