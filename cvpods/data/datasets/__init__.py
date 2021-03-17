# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .citypersons import CityPersonsDataset
from .cityscapes import CityScapesDataset
from .coco import COCODataset
from .crowdhuman import CrowdHumanDataset
from .imagenet import ImageNetDataset
from .lvis import LVISDataset
from .objects365 import Objects365Dataset
from .torchvision_datasets import CIFAR10Dataset, STL10Datasets
from .voc import VOCDataset
from .widerface import WiderFaceDataset

__all__ = [
    "COCODataset",
    "VOCDataset",
    "CityScapesDataset",
    "ImageNetDataset",
    "WiderFaceDataset",
    "LVISDataset",
    "CityPersonsDataset",
    "Objects365Dataset",
    "CrowdHumanDataset",
    "CIFAR10Dataset",
    "STL10Datasets",
]
