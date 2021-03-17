import os.path as osp
from copy import deepcopy
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torchvision.datasets import CIFAR10, STL10

import cvpods
from cvpods.data.base_dataset import BaseDataset
from cvpods.data.registry import DATASETS, PATH_ROUTES

_PREDEFINED_SPLITS_CIFAR10 = {
    "dataset_type": "CIFAR10Dataset",
    "evaluator_type": {"cifar10": "classification"},
    "cifar10": {
        "cifar10_train": ("cifar10", "train"),
        "cifar10_test": ("cifar10", "test"),
    },
}
PATH_ROUTES.register(_PREDEFINED_SPLITS_CIFAR10, "CIFAR10")


@DATASETS.register()
class CIFAR10Dataset(CIFAR10):

    def __init__(self, cfg, dataset_name, transforms, is_train=True, **kwargs):

        self.cfg = cfg
        self.misc = kwargs

        image_root, split = _PREDEFINED_SPLITS_CIFAR10["cifar10"][dataset_name]
        self.data_root = osp.join(osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")

        if is_train:
            assert split == "train"
        else:
            assert split == "test"

        super(CIFAR10Dataset, self).__init__(
            root=osp.join(self.data_root, image_root),
            train=is_train,
            download=True,
            transform=transforms)
        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        self.transforms = self.transform
        self._apply_transforms = BaseDataset._apply_transforms
        self.meta["evaluator_type"] = "classification"

    def __getitem__(self, index):
        image, target = Image.fromarray(self.data[index]), self.targets[index]
        dataset_dict = {"image_id": index, "category_id": target}

        image = image.convert("RGB")
        image = np.asarray(image)
        # flip channels if needed for RGB to BGR
        image = image[:, :, ::-1]

        images, _ = self._apply_transforms(self, image, dataset_dict)

        def process(dd, img):
            if img.shape[0] == 3:  # CHW
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img))
            elif len(img.shape) == 3 and img.shape[-1] == 3:
                dd["image"] = torch.as_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
            elif len(img.shape) == 4 and img.shape[-1] == 3:
                # NHWC -> NCHW
                dd["image"] = torch.as_tensor(
                    np.ascontiguousarray(img.transpose(0, 3, 1, 2)))

            return dd

        if isinstance(images, dict):
            ret = {}
            # multiple input pipelines
            for desc, item in images.items():
                img, anno = item
                ret[desc] = process(deepcopy(dataset_dict), img)
            return ret
        else:
            return process(dataset_dict, images)


_PREDEFINED_SPLITS_STL10 = {
    "dataset_type": "STL10Datasets",
    "evaluator_type": {"stl10": "classification"},
    "stl10": {
        "stl10_train": ("stl10", "train"),
        "stl10_unlabeled": ("stl10", "unlabeled"),
        "stl10_test": ("stl10", "test"),
    },
}
PATH_ROUTES.register(_PREDEFINED_SPLITS_STL10, "STL10")


@DATASETS.register()
class STL10Datasets(STL10):
    def __init__(self, cfg, dataset_name, transforms=[], is_train=True, **kwargs):
        self.cfg = cfg
        self.misc = kwargs

        image_root, split = _PREDEFINED_SPLITS_STL10["stl10"][dataset_name]
        self.data_root = osp.join(osp.split(osp.split(cvpods.__file__)[0])[0], "datasets")
        self.image_root = osp.join(self.data_root, image_root)
        super(STL10Datasets, self).__init__(
            self.image_root, split=split, download=True, transform=None)

        self.aspect_ratios = np.zeros(len(self), dtype=np.uint8)
        self.transforms = transforms
        self._apply_transforms = BaseDataset._apply_transforms

        self.is_train = is_train
        self.meta = {"evaluator_type": _PREDEFINED_SPLITS_STL10["evaluator_type"]["stl10"]}

    def __getitem__(self, index):

        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(np.transpose(img, (1, 2, 0)))
        dataset_dict = {"image_id": index, "category_id": target}

        # format == BGR in cvpods
        image = image.convert("RGB")
        image = np.asarray(image)
        image = image[:, :, ::-1]  # flip channels for RGB -> BGR format

        images, _ = self._apply_transforms(self, image, dataset_dict)

        def process(dd, img):
            if img.shape[0] == 3:  # CHW
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img))
            if len(img.shape) == 3 and img.shape[-1] == 3:
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            elif len(img.shape) == 4 and img.shape[-1] == 3:
                # NHWC -> NCHW
                dd["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(0, 3, 1, 2)))

            return dd

        if isinstance(images, dict):
            ret = {}
            # multiple input pipelines
            for desc, item in images.items():
                img, anno = item
                ret[desc] = process(deepcopy(dataset_dict), img)
            return ret
        else:
            return process(dataset_dict, images)
