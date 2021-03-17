#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.

import copy
import logging
import os
import os.path as osp

import numpy as np

import torch

from cvpods.structures import BoxMode
from cvpods.utils import PathManager, Timer

from ..base_dataset import BaseDataset
from ..detection_utils import (
    annotations_to_instances,
    check_image_size,
    create_keypoint_hflip_indices,
    filter_empty_instances,
    read_image
)
from ..registry import DATASETS
from .lvis_categories import LVIS_CATEGORIES
from .paths_route import _PREDEFINED_SPLITS_LVIS

"""
This file contains functions to parse LVIS-format annotations into dicts in "cvpods format".
"""

logger = logging.getLogger(__name__)


@DATASETS.register()
class LVISDataset(BaseDataset):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        super(LVISDataset, self).__init__(cfg, dataset_name, transforms, is_train)

        assert (
            self.name.startswith("lvis_v0.5")
        ), "Only lvis_v0.5 is now supported, lvis_v1 will be supported in the future."

        image_root, json_file = _PREDEFINED_SPLITS_LVIS["lvis_v0.5"][self.name]
        self.json_file = osp.join(self.data_root, json_file) \
            if "://" not in image_root else osp.join(image_root, json_file)
        self.image_root = osp.join(self.data_root, image_root) \
            if "://" not in image_root else image_root

        self.meta = self._get_metadata()

        self.dataset_dicts = self._load_annotations(
            self.json_file,
            self.image_root)

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

    def __reset__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset_dicts)

    def _load_annotations(self,
                          json_file,
                          image_root):
        """
        Load a json file in LVIS's annotation format.
        Args:
            json_file (str): full path to the LVIS json annotation file.
            image_root (str): the directory where the images in this json file exists.
        Returns:
            list[dict]: a list of dicts in cvpods standard format. (See
            `Using Custom Datasets </tutorials/datasets.html>`_ )
        Notes:
            1. This function does not read the image files.
            The results do not have the "image" field.
        """
        from lvis import LVIS

        json_file = PathManager.get_local_path(json_file)

        timer = Timer()
        lvis_api = LVIS(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        # sort indices for reproducible results
        img_ids = sorted(lvis_api.imgs.keys())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = lvis_api.load_imgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

        # Sanity check that each annotation has a unique id
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
            json_file
        )

        imgs_anns = list(zip(imgs, anns))

        logger.info("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file))

        dataset_dicts = []
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                # Convert form the COCO 2014 file naming convention of
                # COCO_[train/val/test]2014_000000000000.jpg to the 2017 naming convention of
                # 000000000000.jpg (LVIS v1 will fix this naming issue)
                file_name = file_name[-16:]
            record["file_name"] = os.path.join(image_root, file_name)
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
            record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.
                assert anno["image_id"] == image_id
                obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
                obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
                segm = anno["segmentation"]  # list[list[float]]
                # filter out invalid polygons (< 3 points)
                valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                assert len(segm) == len(
                    valid_segm
                ), "Annotation contains an invalid polygon with < 3 points"
                assert len(segm) > 0
                obj["segmentation"] = segm
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        return dataset_dicts

    def _get_metadata(self):
        if "lvis_v0.5" in self.name:
            assert len(LVIS_CATEGORIES) == 1230
            cat_ids = [k["id"] for k in LVIS_CATEGORIES]
            assert min(cat_ids) == 1 and max(cat_ids) == len(
                cat_ids
            ), "Category ids are not in [1, #categories], as expected"
            # Ensure that the category list is sorted by id
            lvis_categories = sorted(LVIS_CATEGORIES, key=lambda x: x["id"])
            thing_classes = [k["synonyms"][0] for k in lvis_categories]
            meta = {
                "thing_classes": thing_classes
            }
        # There will be a v1 in the future
        # elif "lvis_v1" in self.name:
        #   return _get_lvis_instances_meta_v1()
        else:
            raise ValueError("No built-in metadata for dataset {}.".format(self.name))
        meta["evaluator_type"] = _PREDEFINED_SPLITS_LVIS["evaluator_type"]["lvis_v0.5"]
        meta["image_root"] = self.image_root
        meta["json_file"] = self.json_file
        return meta

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        return self.dataset_dicts
