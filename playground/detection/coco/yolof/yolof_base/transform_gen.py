#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import inspect
import pprint
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
from PIL import Image

from cvpods.data.registry import TRANSFORMS

from .transform import (  # isort:skip
    JitterCropTransform,
    HFlipTransform,
    VFlipTransform,
    NoOpTransform,
    DistortTransform2,
    ShiftTransform,
    ResizeTransform,
)

__all__ = [
    "JitterCropYOLOF",
    "RandomDistortionYOLOF",
    "RandomShiftYOLOF",
    "RandomFlipYOLOF",
    "ResizeYOLOF",
    "TransformGenYOLOF",
]


def check_dtype(img):
    """
    Check the image data type and dimensions to ensure that transforms can be applied on it.

    Args:
        img (np.array): image to be checked.
    """
    assert isinstance(
        img, np.ndarray
    ), "[TransformGenYOLOF] Needs an numpy array, but got a {}!".format(type(img))
    assert not isinstance(img.dtype, np.integer) or (
            img.dtype == np.uint8
    ), "[TransformGenYOLOF] Got image of type {}, use uint8 or floating points instead!".format(
        img.dtype
    )
    assert img.ndim in [2, 3], img.ndim


@TRANSFORMS.register()
class TransformGenYOLOF(metaclass=ABCMeta):
    """
    TransformGenYOLOF takes an image of type uint8 in range [0, 255], or
    floating point in range [0, 1] or [0, 255] as input.

    It creates a :class:`Transform` based on the given image, sometimes with randomness.
    The transform can then be used to transform images
    or other data (boxes, points, annotations, etc.) associated with it.

    The assumption made in this class
    is that the image itself is sufficient to instantiate a transform.
    When this assumption is not true, you need to create the transforms by your own.

    A list of `TransformGenYOLOF` can be applied with :func:`apply_transform_gens`.
    """

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def get_transform(self, img, annotations=None):
        raise NotImplementedError

    def __call__(self, img, annotations=None, **kwargs):
        return self.get_transform(img, annotations)(img, annotations, **kwargs)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def __repr__(self):
        """
        Produce something like:
        "MyTransformGenYOLOF(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                        param.kind != param.VAR_POSITIONAL
                        and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name
                    )
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                argstr.append("{}={}".format(name, pprint.pformat(attr)))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__



@TRANSFORMS.register()
class RandomFlipYOLOF(TransformGenYOLOF):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead."
            )
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._init(locals())

    def get_transform(self, img, annotations=None):
        h, w = img.shape[:2]
        do = self._rand_range() < self.prob
        if do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()



@TRANSFORMS.register()
class RandomShiftYOLOF(TransformGenYOLOF):
    """
    Shift the image and box given shift pixels and probability.
    """

    def __init__(self, prob=0.5, max_shifts=8):
        """
        Args:
            prob (float): probability of shifts.
            max_shifts (int): the max pixels for shifting.
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        do = self._rand_range() < self.prob
        if do:
            shift_x = np.random.randint(low=-self.max_shifts,
                                        high=self.max_shifts)
            shift_y = np.random.randint(low=-self.max_shifts,
                                        high=self.max_shifts)
            return ShiftTransform(shift_x, shift_y)
        else:
            return NoOpTransform()


@TRANSFORMS.register()
class JitterCropYOLOF(TransformGenYOLOF):
    """Jitter and crop the image and box."""

    def __init__(self, jitter_ratio):
        super().__init__()
        self._init(locals())

    def __call__(self, img, annotations=None, **kwargs):
        for annotation in annotations:
            annotation["meta_infos"] = dict()
        return self.get_transform(img, annotations)(img, annotations)

    def get_transform(self, img, annotations=None):
        oh, ow = img.shape[:2]
        dw = int(ow * self.jitter_ratio)
        dh = int(oh * self.jitter_ratio)
        pleft = np.random.randint(-dw, dw)
        pright = np.random.randint(-dw, dw)
        ptop = np.random.randint(-dh, dh)
        pbot = np.random.randint(-dh, dh)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot
        return JitterCropTransform(
            pleft=pleft, pright=pright, ptop=ptop, pbot=pbot,
            output_size=(swidth, sheight))


@TRANSFORMS.register()
class RandomDistortionYOLOF(TransformGenYOLOF):
    """
    Random distort image's hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        """
        RandomDistortion Initialization.
        Args:
            hue (float): value of hue
            saturation (float): value of saturation
            exposure (float): value of exposure
        """
        super().__init__()
        self._init(locals())

    def get_transform(self, img, annotations=None):
        return DistortTransform2(self.hue, self.saturation, self.exposure)

@TRANSFORMS.register()
class ResizeYOLOF(TransformGenYOLOF):
    """
    Resize image to a target size
    """

    def __init__(self, shape, interp=Image.BILINEAR, scale_jitter=None):
        """
        Args:
            shape: (h, w) tuple or a int.
            interp: PIL interpolation method.
            scale_jitter: None or (0.8, 1.2)
        """
        if isinstance(shape, int):
            shape = (shape, shape)
        shape = tuple(shape)
        assert (scale_jitter is None or isinstance(scale_jitter, tuple))
        self._init(locals())

    def get_transform(self, img, annotations=None):
        if self.scale_jitter is not None:
            if len(self.scale_jitter) > 2:
                assert isinstance(self.scale_jitter[0], tuple)
                idx = np.random.choice(range(len(self.scale_jitter)))
                shape = self.scale_jitter[idx]
            else:
                jitter = np.random.uniform(self.scale_jitter[0], self.scale_jitter[1])
                shape = (int(self.shape[0] * jitter), int(self.shape[1] * jitter))
        else:
            shape = self.shape
        return ResizeTransform(
            img.shape[0], img.shape[1], shape[0], shape[1], self.interp
        )

