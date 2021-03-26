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

from cvpods.data.transforms.transform_util import to_float_tensor, to_numpy

__all__ = [
    "JitterCropTransform",
    "HFlipTransform",
    "VFlipTransform",
    "NoOpTransform",
    "DistortTransform2",
    "ShiftTransform",
    "Transform",
    "ResizeTransform",
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
