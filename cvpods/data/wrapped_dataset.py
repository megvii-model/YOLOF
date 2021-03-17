#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from types import SimpleNamespace

import numpy as np

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .registry import DATASETS


@DATASETS.register()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.
    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.
    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        if hasattr(self.datasets[0], 'aspect_ratios'):
            aspect_ratios = [d.aspect_ratios for d in self.datasets]
            self.aspect_ratios = np.concatenate(aspect_ratios)
        if hasattr(self.datasets[0], 'meta'):
            self.meta = {}
            for d in self.datasets:
                self.meta.update(d.meta)
            self.meta = SimpleNamespace(**self.meta)


@DATASETS.register()
class RepeatDataset(object):
    """A wrapper of repeated dataset.
    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.
    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        if hasattr(self.dataset, 'aspect_ratios'):
            self.aspect_ratios = np.tile(self.dataset.aspect_ratios, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
