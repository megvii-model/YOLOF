#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import itertools
import logging
import os.path as osp
import pickle
from copy import deepcopy
from tabulate import tabulate
from termcolor import colored

import numpy as np

from torch.utils.data import Dataset

import cvpods
from cvpods.data.detection_utils import check_metadata_consistency, read_image
from cvpods.structures import BoxMode
from cvpods.utils import PathManager, log_first_n

from .registry import DATASETS


@DATASETS.register()
class BaseDataset(Dataset):
    """Abstract class representing a pytorch-like Dataset.
    All other datasets should be subclasses of it.
    All subclasses should override:
        ``__len__`` that provides the size of the dataset,
        ``__getitem__`` that supports integer indexing in the range from 0 to length,
        ``_get_metadata`` that stores dataset meta such as category lists,
        ``_apply_transforms`` that specifies how to apply transformation onto data,
        ``_load_annotations`` that specfies how to access label files,
        ``evaluate`` that is responsible for evaluate predictions of this dataset.

    Default annotation type:
    [
        {
            'file_name': 'a.jpg',
            'width': 1280,
            'height': 720,
            'image_id': if necessary
            'annotations': {
                'bboxes': <np.ndarray> (n, 4),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'masks': polygon or mle (optional)
                'semantic_seg': xxx (optional)
                'labels': <np.ndarray> (n, ), (optional)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]
    """

    def __init__(self, cfg, dataset_name, transforms=[], is_train=True):
        """
        BaseDataset should have the following properties:
            * data_root (contains data and annotations)
            * transforms list
            * evaluators list

        Args:
            cfg (BaseConfig): config
            dataset_name (str): name of the dataset
            transforms (List[TransformGen]): list of transforms to get network input.
            is_train (bool): whether in training mode.
        """
        super(BaseDataset, self).__init__()

        self.name = dataset_name
        self.data_root = osp.join(
            osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")
        self.transforms = transforms
        self.is_train = is_train

        self.data_format = cfg.INPUT.FORMAT

    def __getitem__(self, index):
        """Load data, apply transforms, converto to Instances.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _load_annotations(self):
        raise NotImplementedError

    def _get_metadata(self):
        raise NotImplementedError

    def _read_data(self, file_name):
        return read_image(file_name, format=self.data_format)

    def _apply_transforms(self, image, annotations=None, **kwargs):
        """
        Apply a list of :class:`TransformGen` on the input image, and
        returns the transformed image and a list of transforms.

        We cannot simply create and return all transforms without
        applying it to the image, because a subsequent transform may
        need the output of the previous one.

        Args:
            transform_gens (list): list of :class:`TransformGen` instance to
                be applied.
            img (ndarray): uint8 or floating point images with 1 or 3 channels.
            annotations (list): annotations
        Returns:
            ndarray: the transformed image
            TransformList: contain the transforms that's used.
        """

        if isinstance(self.transforms, dict):
            dataset_dict = {}
            for key, tfms in self.transforms.items():
                img = deepcopy(image)
                annos = deepcopy(annotations)
                for tfm in tfms:
                    img, annos = tfm(img, annos, **kwargs)
                dataset_dict[key] = (img, annos)
            return dataset_dict, None
        else:
            for tfm in self.transforms:
                image, annotations = tfm(image, annotations, **kwargs)

            return image, annotations

    def _filter_annotations(self, filter_empty=True, min_keypoints=0, proposal_files=None):
        """
        Load and prepare dataset dicts for instance detection/segmentation and
        semantic segmentation.

        Args:
            dataset_names (list[str]): a list of dataset names
            filter_empty (bool): whether to filter out images without instance annotations
            min_keypoints (int): filter out images with fewer keypoints than
                `min_keypoints`. Set to 0 to do nothing.
            proposal_files (list[str]): if given, a list of object proposal files
                that match each dataset in `dataset_names`.
        """
        dataset_dicts = self.dataset_dicts

        if proposal_files is not None:
            assert len(self.name) == len(proposal_files)
            # load precomputed proposals from proposal files
            dataset_dicts = [
                load_proposals_into_dataset(dataset_i_dicts, proposal_file)
                for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
            ]

        has_instances = "annotations" in dataset_dicts[0]
        # Keep images without instance-level GT if the dataset has semantic labels
        # unless the task is panoptic segmentation.
        if filter_empty and has_instances:
            dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)

        if min_keypoints > 0 and has_instances:
            dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

        if has_instances:
            try:
                class_names = self.meta["thing_classes"]
                check_metadata_consistency("thing_classes", self.name, self.meta)
                print_instances_class_histogram(dataset_dicts, class_names)
            except AttributeError:  # class names are not available for this dataset
                pass

        return dataset_dicts

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        if "width" in self.dataset_dicts[0] and "height" in self.dataset_dicts[0]:
            for i in range(len(self)):
                dataset_dict = self.dataset_dicts[i]
                if dataset_dict['width'] / dataset_dict['height'] > 1:
                    self.aspect_ratios[i] = 1

    def evaluate(self, predictions):
        """Dataset must provide a evaluation function to evaluate model."""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        raise NotImplementedError


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in cvpods Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in cvpods Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts
        if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info("Removed {} images with fewer than {} keypoints.".format(
        num_before - num_after, min_keypoints_per_image))
    return dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    r"""
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in cvpods Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {
        str(id): i
        for i, id in enumerate(proposals["ids"]) if str(id) in img_ids
    }

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes, ), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = [x["category_id"] for x in annos if not x.get("iscrowd", 0)]
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(
            *[[short_name(class_names[i]), int(v)]
              for i, v in enumerate(histogram)]))
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(
            num_classes) + colored(table, "cyan"),
        key="message",
    )
