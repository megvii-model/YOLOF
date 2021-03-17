# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat

from cvpods import _C
from cvpods.layers.rotated_boxes import pairwise_iou_rotated
from cvpods.utils.apex_wrapper import float_function

ml_nms = _C.ml_nms


@float_function
def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def batched_softnms(boxes, scores, idxs, iou_threshold,
                    score_threshold=0.001, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]
    assert boxes.shape[-1] == 4

    # change scores inplace
    # no need to return changed scores
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        scores[mask] = softnms(boxes[mask], scores[mask], iou_threshold,
                               score_threshold, soft_mode)

    keep = (scores > score_threshold).nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def batched_softnms_rotated(boxes, scores, idxs, iou_threshold,
                            score_threshold=0.001, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]
    assert boxes.shape[-1] == 5

    # change scores inplace
    # no need to return changed scores
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        scores[mask] = softnms_rotated(boxes[mask], scores[mask], iou_threshold,
                                       score_threshold, soft_mode)

    keep = (scores > score_threshold).nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def generalized_batched_nms(boxes, scores, idxs, iou_threshold,
                            score_threshold=0.001, nms_type="normal"):
    assert boxes.shape[-1] == 4

    if nms_type == "normal":
        keep = batched_nms(boxes, scores, idxs, iou_threshold)
    elif nms_type.startswith("softnms"):
        keep = batched_softnms(boxes, scores, idxs, iou_threshold,
                               score_threshold=score_threshold,
                               soft_mode=nms_type.lstrip("softnms-"))
    elif nms_type == "cluster":
        keep = batched_clusternms(boxes, scores, idxs, iou_threshold)
    else:
        raise NotImplementedError("NMS type not implemented: \"{}\"".format(nms_type))

    return keep


def iou(boxes, top_box):
    x1 = boxes[:, 0].clamp(min=top_box[0])
    y1 = boxes[:, 1].clamp(min=top_box[1])
    x2 = boxes[:, 2].clamp(max=top_box[2])
    y2 = boxes[:, 3].clamp(max=top_box[3])

    inters = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    unions = (top_box[2] - top_box[0]) * (top_box[3] - top_box[1]) + areas - inters

    return inters / unions


def scale_by_iou(ious, sigma, soft_mode="gaussian"):
    if soft_mode == "linear":
        scale = ious.new_ones(ious.size())
        scale[ious >= sigma] = 1 - ious[ious >= sigma]
    else:
        scale = torch.exp(-ious ** 2 / sigma)

    return scale


def softnms(boxes, scores, sigma, score_threshold, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]

    undone_mask = scores >= score_threshold
    while undone_mask.sum() > 1:
        idx = scores[undone_mask].argmax()
        idx = undone_mask.nonzero(as_tuple=False)[idx].item()
        top_box = boxes[idx]
        undone_mask[idx] = False
        _boxes = boxes[undone_mask]

        ious = iou(_boxes, top_box)
        scales = scale_by_iou(ious, sigma, soft_mode)

        scores[undone_mask] *= scales
        undone_mask[scores < score_threshold] = False
    return scores


def softnms_rotated(boxes, scores, sigma, score_threshold, soft_mode="gaussian"):
    assert soft_mode in ["linear", "gaussian"]

    iou_matrix = pairwise_iou_rotated(boxes, boxes)

    undone_mask = scores >= score_threshold
    while undone_mask.sum() > 1:
        idx = scores[undone_mask].argmax()
        idx = undone_mask.nonzero(as_tuple=False)[idx].item()
        undone_mask[idx] = False

        ious = iou_matrix[idx, undone_mask]
        scales = scale_by_iou(ious, sigma, soft_mode)

        scores[undone_mask] *= scales
        undone_mask[scores < score_threshold] = False
    return scores


def batched_clusternms(boxes, scores, idxs, iou_threshold):
    assert boxes.shape[-1] == 4

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero(as_tuple=False).view(-1)
        keep = cluster_nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero(as_tuple=False).view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


def cluster_nms(boxes, scores, iou_threshold):
    last_keep = torch.ones(*scores.shape).to(boxes.device)

    scores, idx = scores.sort(descending=True)
    boxes = boxes[idx]
    origin_iou_matrix = box_ops.box_iou(boxes, boxes).tril(diagonal=-1).transpose(1, 0)

    while True:
        iou_matrix = torch.mm(torch.diag(last_keep.float()), origin_iou_matrix)
        keep = (iou_matrix.max(dim=0)[0] <= iou_threshold)

        if (keep == last_keep).all():
            return idx[keep.nonzero(as_tuple=False)]

        last_keep = keep


# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes, scores, iou_threshold):
    r"""
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    from cvpods import _C

    return _C.nms_rotated(boxes, scores, iou_threshold)


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.min(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel="gaussian", sigma=2.0, sum_masks=None):
    """
    Matrix NMS for multi-class masks.
    See: https://arxiv.org/pdf/2003.10152.pdf for more details.

    Args:
        seg_masks (Tensor): shape: [N, H, W], binary masks.
        cate_labels (Tensor): shepe: [N], mask labels in descending order.
        cate_scores (Tensor): shape [N], mask scores in descending order.
        kernel (str):  'linear' or 'gaussian'.
        sigma (float): std in gaussian method.
        sum_masks (Tensor): The sum of seg_masks.

    Returns:
        Tensor: cate_scores_update, tensors of shape [N].
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (
        inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == "gaussian":
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update
