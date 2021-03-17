#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import inspect
import random
from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pycocotools.mask as mask_util

import torch
import torchvision.transforms as transforms

import cvpods
from cvpods.structures import BoxMode

from .transform_util import to_float_tensor, to_numpy

__all__ = [
    "ExpandTransform",
    "AffineTransform",
    "BlendTransform",
    "IoUCropTransform",
    "CropTransform",
    "CropPadTransform",
    "JitterCropTransform",
    "GridSampleTransform",
    "RotationTransform",
    "HFlipTransform",
    "VFlipTransform",
    "NoOpTransform",
    "ScaleTransform",
    "DistortTransform",
    "DistortTransform2",
    "ShiftTransform",
    "Transform",
    "TransformList",
    "ExtentTransform",
    "ResizeTransform",
    # Transform used in ssl
    "GaussianBlurTransform",
    "GaussianBlurConvTransform",
    "SolarizationTransform",
    "ComposeTransform",
    "LabSpaceTransform",
    "PadTransform",
]


# NOTE: to document methods in subclasses, it's sufficient to only document those whose
# implemenation needs special attention.


class Transform(metaclass=ABCMeta):
    """
    Base class for implementations of __deterministic__ transformations for
    image and other data structures. "Deterministic" requires that the output of
    all methods of this class are deterministic w.r.t their input arguments. In
    training, there should be a higher-level policy that generates (likely with
    random variations) these transform ops. Each transform op may handle several
    data types, e.g.: image, coordinates, segmentation, bounding boxes. Some of
    them have a default implementation, but can be overwritten if the default
    isn't appropriate. The implementation of each method may choose to modify
    its input data in-place for efficient transformation.
    """

    def _set_attributes(self, params: list = None):
        """
        Set attributes from the input list of parameters.

        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: image after apply the transformation.
        """
        pass

    @abstractmethod
    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: coordinates after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """

        pass

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.

        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array.
        By default will just transform all the points.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    def apply_meta_infos(self, meta_infos: dict) -> dict:
        return meta_infos

    def __call__(self, image, annotations=None, **kwargs):
        """
        Apply transfrom to images and annotations (if exist)
        """
        image_size = image.shape[:2]  # h, w
        image = self.apply_image(image)

        if annotations is not None:
            for annotation in annotations:
                if "bbox" in annotation:
                    bbox = BoxMode.convert(
                        annotation["bbox"], annotation["bbox_mode"],
                        BoxMode.XYXY_ABS)
                    # Note that bbox is 1d (per-instance bounding box)
                    annotation["bbox"] = self.apply_box([bbox])[0]
                    annotation["bbox_mode"] = BoxMode.XYXY_ABS

                if "segmentation" in annotation:
                    # each instance contains 1 or more polygons
                    segm = annotation["segmentation"]
                    if isinstance(segm, list):
                        # polygons
                        polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
                        annotation["segmentation"] = [
                            p.reshape(-1) for p in
                            self.apply_polygons(polygons)
                        ]
                    elif isinstance(segm, dict):
                        # RLE
                        mask = mask_util.decode(segm)
                        mask = self.apply_segmentation(mask)
                        assert tuple(mask.shape[:2]) == image_size
                        annotation["segmentation"] = mask
                    else:
                        raise ValueError(
                            "Cannot transform segmentation of type '{}'!"
                            "Supported types are: polygons as list[list[float] or ndarray],"
                            " COCO-style RLE as a dict.".format(type(segm)))

                if "keypoints" in annotation:
                    """
                    Transform keypoint annotation of an image.

                    Args:
                        keypoints (list[float]): Nx3 float in cvpods Dataset format.
                        transforms (TransformList):
                        image_size (tuple): the height, width of the transformed image
                        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
                    """
                    # (N*3,) -> (N, 3)
                    keypoints = annotation["keypoints"]
                    keypoints = np.asarray(keypoints, dtype="float64").reshape(
                        -1, 3)
                    keypoints[:, :2] = self.apply_coords(keypoints[:, :2])

                    # This assumes that HorizFlipTransform is the only one that does flip
                    do_hflip = isinstance(self,
                                          cvpods.data.transforms.transform.HFlipTransform)

                    # Alternative way: check if probe points was horizontally flipped.
                    # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
                    # probe_aug = transforms.apply_coords(probe.copy())
                    # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

                    # If flipped, swap each keypoint with its opposite-handed equivalent
                    if do_hflip:
                        if "keypoint_hflip_indices" in kwargs:
                            keypoints = keypoints[
                                        kwargs["keypoint_hflip_indices"], :]

                    # Maintain COCO convention that if visibility == 0, then x, y = 0
                    # TODO may need to reset visibility for cropped keypoints,
                    # but it does not matter for our existing algorithms
                    keypoints[keypoints[:, 2] == 0] = 0

                    annotation["keypoints"] = keypoints

                # For sem seg task
                if "sem_seg" in annotation:
                    sem_seg = annotation["sem_seg"]
                    if isinstance(sem_seg, np.ndarray):
                        sem_seg = self.apply_segmentation(sem_seg)
                        assert tuple(sem_seg.shape[:2]) == tuple(
                            image.shape[:2]), (
                            f"Image shape is {image.shape[:2]}, "
                            f"but sem_seg shape is {sem_seg.shape[:2]}."
                        )
                        annotation["sem_seg"] = sem_seg
                    else:
                        raise ValueError(
                            "Cannot transform segmentation of type '{}'!"
                            "Supported type is ndarray.".format(type(sem_seg)))

                if "meta_infos" in annotation:
                    meta_infos = annotation["meta_infos"]
                    meta_infos = self.apply_meta_infos(meta_infos)
                    annotation["meta_infos"] = meta_infos
        return image, annotations

    @classmethod
    def register_type(cls, data_type: str, func: Callable):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.

        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.

        Examples:

        .. code-block:: python

            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)

            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(
            func)
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(
                str(argspec)))
        setattr(cls, "apply_" + data_type, func)


_T = TypeVar("_T")


class ComposeTransform(object):
    """
    Composes several transforms together.
    """

    def __init__(self, tfms):
        """
        Args:
            transforms (list[Transform]): list of transforms to compose.
        """
        super().__init__()
        self.transforms = tfms

    def __eq__(self, other):
        if not isinstance(other, ComposeTransform):
            return False
        return self.transforms == other.transforms

    def __call__(self, img, annotations=None, **kwargs):
        for tfm in self.transforms:
            img, annotations = tfm(img, annotations, **kwargs)
        return img, annotations

    def __repr__(self):
        return "".join([tfm for tfm in self.transforms])


# TODO: Deprecated
# pyre-ignore-all-errors
class TransformList:
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: list):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        for t in transforms:
            assert isinstance(t, Transform), t
        self.transforms = transforms

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattr__(self, name: str):
        """
        Args:
            name (str): name of the attribute.
        """
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        raise AttributeError(
            "TransformList object has no attribute {}".format(name))

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        return TransformList(others + self.transforms)

    def insert(self, idx: int, other: "TransformList") -> "TransformList":
        """
        Args:
            idx (int): insert position.
            other (TransformList): transformation to insert.
        Returns:
            None
        """
        assert idx in range(len(self.transforms))
        others = (other.transforms
                  if isinstance(other, TransformList) else [other])
        self.transforms = self.transforms[:idx] + others + self.transforms[
                                                           idx:]


class DistortTransform(Transform):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure, image_format):
        super().__init__()
        self._set_attributes(locals())
        self.cvt_code = {
            "RGB": (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
            "BGR": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
        }[image_format]
        if saturation > 1.0:
            saturation /= 255.  # in range [0, 1]

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        dtype = img.dtype
        img = cv2.cvtColor(img, self.cvt_code[0])
        img = np.asarray(img, dtype=np.float32) / 255.
        img[:, :, 1] *= dsat
        img[:, :, 2] *= dexp
        H = img[:, :, 0] + dhue

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        img[:, :, 0] = H
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, self.cvt_code[1])
        img = np.asarray(img, dtype=dtype)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation


class JitterCropTransform(Transform):
    """JitterCrop data augmentation used in YOLOv4.

    Notes:
        - Rewrite as Yolo.
        - A different method to crop image

    Steps:
        - 1. get random offset of four boundary
        - 2. get target crop size
        - 3. get target crop image
        - 4. filter bbox by valid region

    Args:
        pleft (int): left offset.
        pright (int): right offset.
        ptop (int): top offset.
        pbot (int): bottom offset.
        output_size (tuple(int)): output size (w, h).
    """

    def __init__(self, pleft, pright, ptop, pbot, output_size):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the cropped image(s).
        """
        oh, ow = img.shape[:2]

        swidth, sheight = self.output_size

        src_rect = [self.pleft, self.ptop, swidth + self.pleft,
                    sheight + self.ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        # rect intersection
        new_src_rect = [max(src_rect[0], img_rect[0]),
                        max(src_rect[1], img_rect[1]),
                        min(src_rect[2], img_rect[2]),
                        min(src_rect[3], img_rect[3])]
        dst_rect = [max(0, -self.pleft),
                    max(0, -self.ptop),
                    max(0, -self.pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -self.ptop) + new_src_rect[3] - new_src_rect[1]]

        # crop the image
        cropped = np.zeros([sheight, swidth, 3], dtype=img.dtype)
        cropped[:, :, ] = np.mean(img, axis=(0, 1))
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            img[new_src_rect[1]:new_src_rect[3],
            new_src_rect[0]:new_src_rect[2]]
        return cropped

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Crop the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        coords_offset = np.array([self.pleft, self.ptop], dtype=np.float32)
        coords = coords - coords_offset
        swidth, sheight = self.output_size
        coords[..., 0] = np.clip(coords[..., 0], 0, swidth - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, sheight - 1)
        return coords

    def apply_meta_infos(self, meta_infos: dict) -> dict:
        meta_infos["jitter_pad_left"] = self.pleft
        meta_infos["jitter_pad_right"] = self.pright
        meta_infos["jitter_pad_top"] = self.ptop
        meta_infos["jitter_pad_bot"] = self.pbot
        meta_infos["jitter_swidth"] = self.output_size[0]
        meta_infos["jitter_sheight"] = self.output_size[1]
        return meta_infos


class DistortTransform2(Transform):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = np.asarray(img, dtype=np.float32) / 255.
        img[:, :, 1] *= dsat
        img[:, :, 2] *= dexp
        H = img[:, :, 0] + dhue * 179 / 255.

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        img[:, :, 0] = H
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.asarray(img, dtype=np.float32)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale


class AffineTransform(Transform):
    """
    Augmentation from CenterNet
    """

    def __init__(self, src, dst, output_size, pad_value=[0, 0, 0]):
        """
        output_size:(w, h)
        """
        super().__init__()
        affine = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img,
                              self.affine,
                              self.output_size,
                              flags=cv2.INTER_LINEAR,
                              borderValue=self.pad_value)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        # aug_coord (N, 3) shape, self.affine (2, 3) shape
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (
            abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h),
                              flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(
                self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array(
                [self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2,
            self.w, self.h
        )
        return TransformList([rotation, crop])


class HFlipTransform(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-1))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-2))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def apply_meta_infos(self, meta_infos: dict) -> dict:
        pleft = meta_infos["jitter_pad_left"]
        pright = meta_infos["jitter_pad_right"]
        pleft, pright = pright, pleft
        meta_infos["jitter_pad_left"] = pleft
        meta_infos["jitter_pad_right"] = pright
        return meta_infos


class VFlipTransform(Transform):
    """
    Perform vertical flip.
    """

    def __init__(self, height: int):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img).copy())
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            tensor = tensor.flip((-2))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            tensor = tensor.flip((-3))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 1] = self.height - coords[:, 1]
        return coords


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class GaussianBlurTransform(Transform):
    """
    GaussianBlur using PIL.ImageFilter.GaussianBlur
    """

    def __init__(self, sigma, p=1.0):
        """
        Args:
            sigma (List(float)): sigma of gaussian
            p (float): probability of perform this augmentation
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = Image.fromarray(img).filter(
                ImageFilter.GaussianBlur(radius=sigma))
        return np.array(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class SolarizationTransform(Transform):
    def __init__(self, thresh=128, p=0.5):
        super().__init__()
        self.thresh = thresh
        self.p = p

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return np.array(
                ImageOps.solarize(Image.fromarray(img), self.thresh))
        else:
            return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class GaussianBlurConvTransform(Transform):
    def __init__(self, kernel_size, p=1.0):
        super().__init__()
        self._set_attributes(locals())
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                      stride=1, padding=0, bias=False,
                                      groups=3)
        self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                      stride=1, padding=0, bias=False,
                                      groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            img = self.pil_to_tensor(Image.fromarray(img)).unsqueeze(0)

            sigma = np.random.uniform(0.1, 2.0)
            x = np.arange(-self.r, self.r + 1)
            x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
            x = x / x.sum()
            x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

            self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
            self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

            with torch.no_grad():
                img = self.blur(img)
                img = img.squeeze()

            img = np.array(self.tensor_to_pil(img))
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class LabSpaceTransform(Transform):
    """
    Convert image from RGB into Lab color space
    """

    def __init__(self):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        assert len(img.shape) == 3, 'Image should have dim H x W x 3'
        assert img.shape[2] == 3, 'Image should have dim H x W x 3'
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab = img_lab.astype(np.float32)
        img_lab[:, :, 0] = (img_lab[:, :, 0] * (100.0 / 255.0)) - 50.0
        img_lab[:, :, 1:] = img_lab[:, :, 1:] - 128.0
        return img_lab

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords


class PadTransform(Transform):
    """
    Pad image with `pad_value` to the specified `target_h` and `target_w`.

    Adds `top` rows of `pad_value` on top, `left` columns of `pad_value` on the left,
    and then pads the image on the bottom and right with `pad_value` until it has
    dimensions `target_h`, `target_w`.

    This op does nothing if `top` and `left` is zero and the image already has size
    `target_h` by `target_w`.
    """

    def __init__(self,
                 top: int,
                 left: int,
                 target_h: int,
                 target_w: int,
                 pad_value=0,
                 seg_value=255,
                 ):
        """
        Args:
            top (int): number of rows of `pad_value` to add on top.
            left (int): number of columns of `pad_value` to add on the left.
            target_h (int): height of output image.
            target_w (int): width of output image.
            pad_value (int): the value used to pad the image.
            seg_value (int): the value used to pad the semantic seg annotaions.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, pad_value=None) -> np.ndarray:
        if pad_value is None:
            pad_value = self.pad_value

        if len(img.shape) == 2:  # semantic segmentation mask
            shape = (self.target_h, self.target_w)
        else:
            shape = (self.target_h, self.target_w, 3)

        pad_img = np.full(shape=shape, fill_value=pad_value).astype(img.dtype)

        rest_h = self.target_h - self.top
        rest_w = self.target_w - self.left

        img_h, img_w = img.shape[:2]
        paste_h, paste_w = min(rest_h, img_h), min(rest_w, img_w)
        pad_img[self.top:self.top + paste_h,
        self.left:self.left + paste_w] = img[:paste_h, :paste_w]
        return pad_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        coords[:, 0] = coords[:, 0] + self.left
        coords[:, 1] = coords[:, 1] + self.top
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply pad transform on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: padded segmentation.
        """
        segmentation = self.apply_image(segmentation, pad_value=self.seg_value)
        return segmentation


class ScaleTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self,
                 h: int,
                 w: int,
                 new_h: int,
                 new_w: int,
                 interp: str = "BILINEAR"):
        """
        Args:
            h, w (int): original image size.
            new_h, new_w (int): new image size.
            interp (str): the interpolation method. Options includes:
              * "NEAREST"
              * "BILINEAR"
              * "BICUBIC"
              * "LANCZOS"
              * "HAMMING"
              * "BOX"
        """
        super().__init__()
        self._set_attributes(locals())
        _str_to_pil_interpolation = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS,
            "HAMMING": Image.HAMMING,
            "BOX": Image.BOX,
        }
        assert (interp in _str_to_pil_interpolation.keys(
        )), "This interpolation mode ({}) is not currently supported!".format(
            interp)
        self.interp = _str_to_pil_interpolation[interp]

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Resize the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: resized image(s).
        """
        # Method 1: second fastest
        # img = cv2.resize(img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)

        # Method 2: fastest
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        img = np.asarray(pil_image)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the coordinates after resize.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: resized coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply resize on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: resized segmentation.
        """
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class GridSampleTransform(Transform):
    def __init__(self, grid: np.ndarray, interp: str):
        """
        Args:
            grid (ndarray): grid has x and y input pixel locations which are
                used to compute output. Grid has values in the range of [-1, 1],
                which is normalized by the input height and width. The dimension
                is `N x H x W x 2`.
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply grid sampling on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        Returns:
            ndarray: grid sampled image(s).
        """
        interp_method = interp if interp is not None else self.interp
        float_tensor = torch.nn.functional.grid_sample(
            to_float_tensor(img),  # NxHxWxC -> NxCxHxW.
            torch.from_numpy(self.grid),
            mode=interp_method,
            padding_mode="border",
            align_corners=False,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray):
        """
        Not supported.
        """
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply grid sampling on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: grid sampled segmentation.
        """
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class IoUCropTransform(Transform):
    """
    Perform crop operations on images.

    This crop operation will checks whether the center of each instance's bbox
    is in the cropped image.
    """

    def __init__(self, x0: int, y0: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        else:
            return img[..., self.y0:self.y0 + self.h,
                   self.x0:self.x0 + self.w, :]

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on an axis-aligned box.
        By default will transform the corner points and use their
        minimum/maximum to create a new axis-aligned box.
        Note that this default may change the size of your box, e.g. in
        rotations.

        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.

        Returns:
            ndarray: box after apply the transformation.

        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        box = np.array(box).reshape(-1, 4)
        center = (box[:, :2] + box[:, 2:]) / 2
        mask = ((center[:, 0] > self.x0) * (center[:, 0] < self.x0 + self.w)
                * (center[:, 1] > self.y0) * (center[:, 1] < self.y0 + self.h))
        if not mask.any():
            return np.zeros_like(box)

        tl = np.array([self.x0, self.y0])
        box[:, :2] = np.maximum(box[:, :2], tl)
        box[:, :2] -= tl

        box[:, 2:] = np.minimum(box[:, 2:],
                                np.array([self.x0 + self.w, self.y0 + self.h]))
        box[:, 2:] -= tl

        return box

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.

        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w,
                                self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped,
                              geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class CropTransform(Transform):
    """
    Perform crop operations on images.
    """

    def __init__(self, x0: int, y0: int, w: int, h: int):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        else:
            return img[..., self.y0:self.y0 + self.h,
                   self.x0:self.x0 + self.w, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.

        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w,
                                self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped,
                              geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class CropPadTransform(Transform):
    def __init__(self,
                 x0: int,
                 y0: int,
                 w: int,
                 h: int,
                 new_w: int,
                 new_h: int,
                 img_value=None,
                 seg_value=None):
        super().__init__()
        self._set_attributes(locals())
        self.crop_trans = CropTransform(x0, y0, w, h)
        pad_top_offset = self.get_pad_offset(h, new_h)
        pad_left_offset = self.get_pad_offset(w, new_w)
        self.pad_trans = PadTransform(
            pad_top_offset, pad_left_offset, new_h, new_w, img_value,
            seg_value)

    def get_pad_offset(self, ori: int, tar: int):
        pad_length = max(tar - ori, 0)
        pad_offset = pad_length // 2
        return pad_offset

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop and Pad the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: cropped and padded image(s).
        """
        img = self.crop_trans.apply_image(img)
        img = self.pad_trans.apply_image(img)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: cropped and padded coordinates.
        """
        coords = self.crop_trans.apply_coords(coords)
        coords = self.pad_trans.apply_coords(coords)
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop and pad transform on a list of polygons, each represented by a Nx2 array.

        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.

        Returns:
            ndarray: cropped and padded polygons.
        """
        polygons = self.crop_trans.apply_polygons(polygons)
        polygons = self.pad_trans.apply_polygons(polygons)
        return polygons

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply crop and pad transform on the full-image segmentation.

        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.

        Returns:
            ndarray: cropped and padded segmentation.
        """
        segmentation = self.crop_trans.apply_segmentation(segmentation)
        segmentation = self.pad_trans.apply_segmentation(segmentation)
        return segmentation


class BlendTransform(Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float,
                 dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.

        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation


class ShiftTransform(Transform):
    """
    Shift the image with random pixels.
    """

    def __init__(self, shift_x: int, shift_y: int):
        """
        Args:
            shift_x (int): the shift pixel for x axis.
            shift_y (int): the shift piexl for y axis.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Shift the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: shifted image(s).
        """
        new_img = np.zeros_like(img)
        if self.shift_x < 0:
            new_x = 0
            orig_x = - self.shift_x
        else:
            new_x = self.shift_x
            orig_x = 0
        if self.shift_y < 0:
            new_y = 0
            orig_y = - self.shift_y
        else:
            new_y = self.shift_y
            orig_y = 0

        if len(img.shape) <= 3:
            img_h, img_w = img.shape[:2]
            new_h = img_h - np.abs(self.shift_y)
            new_w = img_w - np.abs(self.shift_x)
            new_img[new_y:new_y + new_h, new_x:new_x + new_w] = img[
                                                                orig_y:orig_y + new_h,
                                                                orig_x:orig_x + new_w]
            return new_img
        else:
            img_h, img_w = img.shape[1:3]
            new_h = img_h - np.abs(self.shift_y)
            new_w = img_w - np.abs(self.shift_x)
            new_img[..., new_y:new_y + new_h, new_x:new_x + new_w, :] = img[
                                                                        ...,
                                                                        orig_y:orig_y + new_h,
                                                                        orig_x:orig_x + new_w,
                                                                        :]
            return new_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply shift transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] += self.shift_x
        coords[:, 1] += self.shift_y
        return coords


class RandomSwapChannelsTransform(Transform):
    """
    Randomly swap image channels.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img):
        assert len(img.shape) > 2
        return img[..., np.random.permutation(3)]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation


class ExpandTransform(Transform):
    """
    Expand the image and boxes according the specified expand ratio.
    """

    def __init__(self, left, top, ratio, mean=(0, 0, 0)):
        """
        Args:
            left, top (int): crop the image by img[top: top+h, left:left+w].
            ratio (float): image expand ratio.
            mean (tuple): mean value of dataset.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        """
        Randomly place the original image on a canvas of 'ratio' x original image
        size filled with mean values. The ratio is in the range of ratio_range.
        """
        h, w, c = img.shape
        expand_img = np.full((int(h * self.ratio), int(w * self.ratio), c),
                             self.mean).astype(img.dtype)
        expand_img[self.top:self.top + h, self.left:self.left + w] = img
        return expand_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply expand transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            ndarray: expand coordinates.
        """
        coords[:, 0] += self.left
        coords[:, 1] += self.top
        return coords


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.

    The fill color is used to map pixels from the source rect that fall outside
    the source image.

    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        ret = Image.fromarray(img).transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        return np.asarray(ret)

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        pil_image = Image.fromarray(img)
        interp_method = interp if interp is not None else self.interp
        pil_image = pil_image.resize((self.new_w, self.new_h), interp_method)
        ret = np.asarray(pil_image)
        return ret

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


def HFlip_rotated_box(transform, rotated_boxes):
    """
    Apply the horizontal flip transform on rotated boxes.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    # Transform x_center
    rotated_boxes[:, 0] = transform.width - rotated_boxes[:, 0]
    # Transform angle
    rotated_boxes[:, 4] = -rotated_boxes[:, 4]
    return rotated_boxes


def Resize_rotated_box(transform, rotated_boxes):
    """
    Apply the resizing transform on rotated boxes. For details of how these (approximation)
    formulas are derived, please refer to :meth:`RotatedBoxes.scale`.

    Args:
        rotated_boxes (ndarray): Nx5 floating point array of
            (x_center, y_center, width, height, angle_degrees) format
            in absolute coordinates.
    """
    scale_factor_x = transform.new_w * 1.0 / transform.w
    scale_factor_y = transform.new_h * 1.0 / transform.h
    rotated_boxes[:, 0] *= scale_factor_x
    rotated_boxes[:, 1] *= scale_factor_y
    theta = rotated_boxes[:, 4] * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)
    rotated_boxes[:, 2] *= np.sqrt(
        np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
    rotated_boxes[:, 3] *= np.sqrt(
        np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
    rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s,
                                     scale_factor_y * c) * 180 / np.pi

    return rotated_boxes


HFlipTransform.register_type("rotated_box", HFlip_rotated_box)
NoOpTransform.register_type("rotated_box", lambda t, x: x)
ResizeTransform.register_type("rotated_box", Resize_rotated_box)
