#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn


class LabelSmoothCELoss(nn.Module):
    """
    Cross-entrophy loss with label smooth.

    Args:
        epsilon: Smoothing level. Use one-hot label when set to 0, use uniform label when set to 1.
    """
    def __init__(self, epsilon):
        super(LabelSmoothCELoss, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets):
        """
        Args:
            logits: A float tensor of shape: (minibatch, C).
            targets: A float tensor of shape: (minibatch,). Stores the class indices
                    in range `[0, C - 1]`.

        Returns:
            A scalar tensor.
        """
        log_probs = self.logsoftmax(logits)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / logits.shape[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def label_smooth_ce_loss(logits, targets, epsilon):
    """
    Cross-entrophy loss with label smooth.

    Args:
        logits: A float tensor of shape: (minibatch, C).
        targets: A float tensor of shape: (minibatch,). Stores the class indices
                 in range `[0, C - 1]`.
        epsilon: Smoothing level. Use one-hot label when set to 0, use uniform label when set to 1.

    Returns:
        A scalar tensor.
    """
    log_probs = nn.functional.log_softmax(logits, dim=1)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - epsilon) * targets + epsilon / logits.shape[1]
    loss = (-targets * log_probs).mean(0).sum()
    return loss
