#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.

import copy
import logging
import os
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np

import torch

from cvpods.structures import BoxMode
from cvpods.utils import PathManager

from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances,
    check_image_size,
    create_keypoint_hflip_indices,
    filter_empty_instances,
    read_image
)
from ..registry import DATASETS
from .paths_route import _PREDEFINED_SPLITS_VOC

"""
This file contains functions to parse ImageNet-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class VOCDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(VOCDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        image_root, split = _PREDEFINED_SPLITS_VOC["voc"][self.name]
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root
        self.split = split

        self.meta = self._get_metadata()
        self.dataset_dicts = self._load_annotations()

        # fmt: off
        self.data_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.filter_empty = cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        self.proposal_files = cfg.DATASETS.PROPOSAL_FILES_TRAIN
        # fmt: on

        if is_train:
            self.dataset_dicts = self._filter_annotations(
                filter_empty=self.filter_empty,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if self.keypoint_on else 0,
                proposal_files=self.proposal_files if self.load_proposals else None,
            )
            self._set_group_flag()

        self.eval_with_gt = cfg.TEST.get("WITH_GT", False)

        if self.keypoint_on:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = create_keypoint_hflip_indices(
                cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        dataset_dict = copy.deepcopy(self.dataset_dicts[index])

        # read image
        image = read_image(dataset_dict["file_name"], format=self.data_format)
        check_image_size(dataset_dict, image)

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            annotations = [
                ann for ann in annotations if ann.get("iscrowd", 0) == 0]
        else:
            annotations = None

        # apply transfrom
        image, annotations = self._apply_transforms(
            image, annotations)

        if annotations is not None:
            image_shape = image.shape[:2]  # h, w

            instances = annotations_to_instances(
                annotations, image_shape, mask_format=self.mask_format
            )

            # # Create a tight bounding box from masks, useful when image is cropped
            # if self.crop_gen and instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = filter_empty_instances(instances)

        # convert to Instance type
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        # h, w, c -> c, h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        return dataset_dict

    def __len__(self):
        return len(self.dataset_dicts)

    def _get_metadata(self):
        # fmt: off
        thing_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]
        meta = {
            "thing_classes": thing_classes,
            "evaluator_type": _PREDEFINED_SPLITS_VOC["evaluator_type"]["voc"],
            "dirname": self.image_root,
            "split": self.split,
            "year": 2007,
        }
        return meta

    def _load_annotations(self):
        """
        Load Pascal VOC detection annotations to cvpods format.

        Args:
            dirname: Contain "Annotations", "ImageSets", "JPEGImages"
            split (str): one of "train", "test", "val", "trainval"
        """

        dirname = self.image_root
        split = self.split

        with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=np.str)

        dicts = []
        for fileid in fileids:
            anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                # We include "difficult" samples in training.
                # Based on limited experiments, they don't hurt accuracy.
                # difficult = int(obj.find("difficult").text)
                # if difficult == 1:
                # continue
                bbox = obj.find("bndbox")
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                instances.append({
                    "category_id": CLASS_NAMES.index(cls),
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS
                })
            r["annotations"] = instances
            dicts.append(r)

        return dicts


# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
