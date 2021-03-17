#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.

import math
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler

# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        """
        Multi Step LR with warmup

        Args:
            optimizer (torch.optim.Optimizer): optimizer used.
            milestones (list[Int]): a list of increasing integers.
            gamma (float): gamma
            warmup_factor (float): lr = warmup_factor * base_lr
            warmup_iters (int): iters to warmup
            warmup_method (str): warmup method in ["constant", "linear", "burnin"]
            last_epoch(int):  The index of last epoch. Default: -1.
        """
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        if isinstance(optimizer, torch.optim.Optimizer):
            super().__init__(optimizer, last_epoch)
        else:
            super().__init__(optimizer.optim, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        return [
            base_lr * warmup_factor * self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        epoch_iters: int = -1,
    ):
        """
        Cosine LR with warmup

        Args:
            optimizer (Optimizer):  Wrapped optimizer.
            max_iters (int): max num of iters
            warmup_factor (float): warmup factor to compute lr
            warmup_iters (int): warmup iters
            warmup_method (str): warmup method in ["constant", "linear", "burnin"]
            last_epoch: The index of last epoch. Default: -1.
        """
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.epoch_iters = epoch_iters
        if isinstance(optimizer, torch.optim.Optimizer):
            super().__init__(optimizer, last_epoch)
        else:
            super().__init__(optimizer.optim, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor)
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        if self.epoch_iters > 0:
            # epoch wise
            coeff = int(self.last_epoch / self.epoch_iters) / int(
                self.max_iters / self.epoch_iters)
        else:
            # iter wise
            coeff = self.last_epoch / self.max_iters

        return [
            base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * coeff))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class PolyLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        power: float = 0.9,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        """
        Poly LR with warmup
        Args:
            optimizer (torch.optim.Optimizer): optimizer used.
            max_iters (int): max num of iters.
            power (float): power
            warmup_factor (float): lr = warmup_factor * base_lr
            warmup_iters (int): iters to warmup
            warmup_method (str): warmup method in ["constant", "linear", "burnin"]
            last_epoch(int):  The index of last epoch. Default: -1.
        """
        self.max_iters = max_iters
        self.power = power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        if isinstance(optimizer, torch.optim.Optimizer):
            super().__init__(optimizer, last_epoch)
        else:
            super().__init__(optimizer.optim, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # self.last_epoch is used for current iter here.
        return [
            base_lr * warmup_factor * (
                (1 - float(self.last_epoch) / self.max_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(method: str, iter: int, warmup_iters: int,
                               warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    elif method == "burnin":
        return (iter / warmup_iters)**4
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
