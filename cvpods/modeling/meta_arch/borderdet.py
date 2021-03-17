#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from cvpods.layers import ShapeSpec, cat, generalized_batched_nms
from cvpods.layers.border_align import BorderAlign
from cvpods.modeling.box_regression import Shift2BoxTransform
from cvpods.modeling.losses import iou_loss, sigmoid_focal_loss_jit, smooth_l1_loss
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.modeling.postprocessing import detector_postprocess
from cvpods.structures import Boxes, ImageList, Instances, pairwise_iou
from cvpods.utils import comm, log_first_n


def permute_all_cls_and_box_to_N_HWA_K_and_concat(
        box_cls, box_delta, box_center, border_cls, border_delta, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the centerness
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    box_center_flattened = [permute_to_N_HWA_K(x, 1) for x in box_center]

    border_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in border_cls]
    border_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in border_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    box_center = cat(box_center_flattened, dim=1).view(-1, 1)

    border_cls = cat(border_cls_flattened, dim=1).view(-1, num_classes)
    border_delta = cat(border_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta, box_center, border_cls, border_delta


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class BorderDet(nn.Module):
    """
    Implement BorderDet.
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # Loss Parameters:
        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.border_iou_thresh = cfg.MODEL.BORDER.IOU_THRESH
        self.border_bbox_std = cfg.MODEL.BORDER.BBOX_STD
        # Inference Parameters:
        self.score_threshold  = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.topk_candidates = cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST
        self.nms_threshold = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_type = cfg.MODEL.NMS_TYPE
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        # build network
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = BorderHead(cfg, feature_shapes)
        self.shift_generator = cfg.build_shift_generator(cfg, feature_shapes)

        # Matching and Loss
        self.shift2box_transform = Shift2BoxTransform(
            weights=cfg.MODEL.FCOS.BBOX_REG_WEIGHTS)
        self.object_sizes_of_interest = cfg.MODEL.FCOS.OBJECT_SIZES_OF_INTEREST

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        shifts = self.shift_generator(features)
        (
            box_cls,
            box_delta,
            box_center,
            bd_box_cls,
            bd_box_delta,
            bd_based_box
        ) = self.head(features, shifts)

        if self.training:
            (
                gt_classes,
                gt_shifts_reg_deltas,
                gt_centerness,
                gt_border_classes,
                gt_border_shifts_deltas
            ) = self.get_ground_truth(shifts, gt_instances, bd_based_box)
            return self.losses(
                gt_classes,
                gt_shifts_reg_deltas,
                gt_centerness,
                gt_border_classes,
                gt_border_shifts_deltas,
                box_cls,
                box_delta,
                box_center,
                bd_box_cls,
                bd_box_delta,
            )
        else:
            results = self.inference(
                box_cls, box_center, bd_box_cls, bd_box_delta, bd_based_box, images.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
            self,
            gt_classes,
            gt_shifts_deltas,
            gt_centerness,
            gt_classes_border,
            gt_deltas_border,
            pred_class_logits,
            pred_shift_deltas,
            pred_centerness,
            border_box_cls,
            border_bbox_reg,
    ):
        """
        Args:
            For `gt_classes`, `gt_shifts_deltas` and `gt_centerness` parameters, see
                :meth:`BorderDet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of shifts across levels, i.e. sum(Hi x Wi)
            For `pred_class_logits`, `pred_shift_deltas` and `pred_centerness`, see
                :meth:`BorderHead.forward`.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        (
            pred_class_logits,
            pred_shift_deltas,
            pred_centerness,
            border_class_logits,
            border_shift_deltas,
        ) = permute_all_cls_and_box_to_N_HWA_K_and_concat(
            pred_class_logits, pred_shift_deltas, pred_centerness,
            border_box_cls, border_bbox_reg, self.num_classes
        )  # Shapes: (N x R, K) and (N x R, 4), respectively.

        # fcos
        gt_classes = gt_classes.flatten()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4)
        gt_centerness = gt_centerness.view(-1, 1)

        valid_idxs = gt_classes >= 0
        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        num_foreground = comm.all_reduce(num_foreground) / float(comm.get_world_size())
        num_foreground_centerness = gt_centerness[foreground_idxs].sum()
        num_targets = comm.all_reduce(num_foreground_centerness)  / float(comm.get_world_size())

        # logits loss
        loss_cls = sigmoid_focal_loss_jit(
            pred_class_logits[valid_idxs],
            gt_classes_target[valid_idxs],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1.0, num_foreground)

        # regression loss
        loss_box_reg = iou_loss(
            pred_shift_deltas[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            gt_centerness[foreground_idxs],
            box_mode="ltrb",
            loss_type=self.iou_loss_type,
            reduction="sum",
        ) / max(1.0, num_targets)

        # centerness loss
        loss_centerness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_idxs],
            gt_centerness[foreground_idxs],
            reduction="sum",
        ) / max(1.0, num_foreground)

        # borderdet
        gt_classes_border = gt_classes_border.flatten()
        gt_deltas_border = gt_deltas_border.view(-1, 4)

        valid_idxs_border = gt_classes_border >= 0
        foreground_idxs_border = (gt_classes_border >= 0) & (gt_classes_border != self.num_classes)
        num_foreground_border = foreground_idxs_border.sum()

        gt_classes_border_target = torch.zeros_like(border_class_logits)
        gt_classes_border_target[
            foreground_idxs_border, gt_classes_border[foreground_idxs_border]] = 1

        num_foreground_border = (
            comm.all_reduce(num_foreground_border) / float(comm.get_world_size())
        )

        num_foreground_border = max(num_foreground_border, 1.0)
        loss_border_cls = sigmoid_focal_loss_jit(
            border_class_logits[valid_idxs_border],
            gt_classes_border_target[valid_idxs_border],
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_foreground_border

        if foreground_idxs_border.numel() > 0:
            loss_border_reg = (
                smooth_l1_loss(
                    border_shift_deltas[foreground_idxs_border],
                    gt_deltas_border[foreground_idxs_border],
                    beta=0,
                    reduction="sum"
                ) / num_foreground_border
            )
        else:
            loss_border_reg = border_shift_deltas.sum()

        return {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_border_cls": loss_border_cls,
            "loss_border_reg": loss_border_reg,
        }

    @torch.no_grad()
    def get_ground_truth(self, shifts, targets, pre_boxes_list):
        """
        Args:
            shifts (list[list[Tensor]]): a list of N=#image elements. Each is a
                list of #feature level tensors. The tensors contains shifts of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            gt_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.
            gt_centerness (Tensor):
                An float tensor (0, 1) of shape (N, R) whose values in [0, 1]
                storing ground-truth centerness for each shift.
            border_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each shift.
                R is the total number of shifts, i.e. the sum of Hi x Wi for all levels.
                Shifts in the valid boxes are assigned their corresponding label in the
                [0, K-1] range. Shifts in the background are assigned the label "K".
                Shifts in the ignore areas are assigned a label "-1", i.e. ignore.
            border_shifts_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth shift2box transform
                targets (dl, dt, dr, db) that map each shift to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                shift is labeled as foreground.

        """
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []

        border_classes = []
        border_shifts_deltas = []

        for shifts_per_image, targets_per_image, pre_boxes in zip(shifts, targets, pre_boxes_list):
            object_sizes_of_interest = torch.cat([
                shifts_i.new_tensor(size).unsqueeze(0).expand(
                    shifts_i.size(0), -1) for shifts_i, size in zip(
                    shifts_per_image, self.object_sizes_of_interest)
            ], dim=0)

            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)

            gt_boxes = targets_per_image.gt_boxes

            deltas = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes.tensor.unsqueeze(1))

            if self.center_sampling_radius > 0:
                centers = gt_boxes.get_centers()
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, gt_boxes.tensor[:, :2]),
                        torch.min(centers + radius, gt_boxes.tensor[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.shift2box_transform.get_deltas(
                        shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            gt_positions_area = gt_boxes.area().unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.shift2box_transform.get_deltas(
                shifts_over_all_feature_maps, gt_boxes[gt_matched_idxs].tensor)

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                gt_classes_i[positions_min_area == math.inf] = self.num_classes
            else:
                gt_classes_i = torch.zeros_like(gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(gt_classes_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

            # border
            iou = pairwise_iou(Boxes(pre_boxes), gt_boxes)
            (max_iou, argmax_iou) = iou.max(dim=1)
            invalid = max_iou < self.border_iou_thresh
            gt_target = gt_boxes[argmax_iou].tensor

            border_cls_target = targets_per_image.gt_classes[argmax_iou]
            border_cls_target[invalid] = self.num_classes

            border_bbox_std = pre_boxes.new_tensor(self.border_bbox_std)
            pre_boxes_wh = pre_boxes[:, 2:4] - pre_boxes[:, 0:2]
            pre_boxes_wh = torch.cat([pre_boxes_wh, pre_boxes_wh], dim=1)
            border_off_target = (gt_target - pre_boxes) / (pre_boxes_wh * border_bbox_std)

            border_classes.append(border_cls_target)
            border_shifts_deltas.append(border_off_target)

        return (
            torch.stack(gt_classes),
            torch.stack(gt_shifts_deltas),
            torch.stack(gt_centerness),
            torch.stack(border_classes),
            torch.stack(border_shifts_deltas),
        )

    def inference(self, box_cls, box_center, border_cls, border_delta, bd_based_box, image_sizes):
        """
        Arguments:
            box_cls, box_delta, box_center: Same as the output of :meth:`BorderHead.forward`
            shifts (list[list[Tensor]): a list of #images elements. Each is a
                list of #feature level tensor. The tensor contain shifts of this
                image on the specific feature level.
            image_sizes (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        results = []

        box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
        box_center = [permute_to_N_HWA_K(x, 1) for x in box_center]
        border_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in border_cls]
        border_delta = [permute_to_N_HWA_K(x, 4) for x in border_delta]
        # list[Tensor], one per level, each has shape (N, Hi x Wi, K or 4)

        for img_idx, image_size_per_image in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_ctr_per_image = [box_ctr_per_level[img_idx] for box_ctr_per_level in box_center]
            border_cls_per_image = [
                border_cls_per_level[img_idx] for border_cls_per_level in border_cls
            ]
            border_reg_per_image = [
                border_reg_per_level[img_idx] for border_reg_per_level in border_delta
            ]
            bd_based_box_per_image = [
                box_loc_per_level[img_idx] for box_loc_per_level in bd_based_box
            ]

            results_per_image = self.inference_single_image(
                box_cls_per_image, box_ctr_per_image, border_cls_per_image,
                border_reg_per_image, bd_based_box_per_image, tuple(image_size_per_image)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(
            self, box_cls, box_center, border_cls, border_delta, bd_based_box, image_size
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            box_center (list[Tensor]): Same shape as 'box_cls' except that K becomes 1.
            shifts (list[Tensor]): list of #feature levels. Each entry contains
                a tensor, which contains all the shifts for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        border_bbox_std = bd_based_box[0].new_tensor(self.border_bbox_std)

        # Iterate over every feature level
        for box_cls_i, box_ctr_i, bd_box_cls_i, bd_box_reg_i, bd_based_box_i in zip(
                box_cls, box_center, border_cls, border_delta, bd_based_box):
            # (HxWxK,)
            box_cls_i = box_cls_i.sigmoid_()
            box_ctr_i = box_ctr_i.sigmoid_()
            bd_box_cls_i = bd_box_cls_i.sigmoid_()

            predicted_prob = (box_cls_i * box_ctr_i).sqrt()

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold

            predicted_prob = predicted_prob * bd_box_cls_i

            predicted_prob = predicted_prob[keep_idxs]
            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, predicted_prob.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = predicted_prob.sort(descending=True)
            topk_idxs = topk_idxs[:num_topk]

            keep_idxs = keep_idxs.nonzero()
            keep_idxs = keep_idxs[topk_idxs]
            keep_box_idxs = keep_idxs[:, 0]
            classes_idxs = keep_idxs[:, 1]

            predicted_prob = predicted_prob[:num_topk]
            bd_box_reg_i = bd_box_reg_i[keep_box_idxs]
            bd_based_box_i = bd_based_box_i[keep_box_idxs]

            det_wh = (bd_based_box_i[..., 2:4] - bd_based_box_i[..., :2])
            det_wh = torch.cat([det_wh, det_wh], dim=1)
            predicted_boxes = bd_based_box_i + (bd_box_reg_i * border_bbox_std * det_wh)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob.sqrt())
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        keep = generalized_batched_nms(boxes_all, scores_all, class_idxs_all,
                                       self.nms_threshold, nms_type=self.nms_type)
        boxes_all = boxes_all[keep]
        scores_all = scores_all[keep]
        class_idxs_all = class_idxs_all[keep]

        number_of_detections = len(keep)
        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.max_detections_per_image > 0:
            image_thresh, _ = torch.kthvalue(
                scores_all,
                number_of_detections - self.max_detections_per_image + 1
            )
            keep = scores_all >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            boxes_all = boxes_all[keep]
            scores_all = scores_all[keep]
            class_idxs_all = class_idxs_all[keep]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all)
        result.scores = scores_all
        result.pred_classes = class_idxs_all
        return result

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class BorderHead(nn.Module):
    """
    The head used in BorderDet for object classification and box regression.
    It has two subnets for the two tasks, with a common structure but separate parameters.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # fmt: on

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            cls_subnet.append(nn.GroupNorm(32, in_channels))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            bbox_subnet.append(nn.GroupNorm(32, in_channels))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1)

        self.add_module("border_cls_subnet", BorderBranch(in_channels, 256))
        self.add_module("border_bbox_subnet", BorderBranch(in_channels, 128))

        self.border_cls_score = nn.Conv2d(
            in_channels, num_classes, kernel_size=1, stride=1)
        self.border_bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=1, stride=1)

        # Initialization
        for modules in [
            self.cls_subnet, self.bbox_subnet,
            self.cls_score, self.bbox_pred, self.centerness,
            self.border_cls_subnet, self.border_bbox_subnet,
            self.border_cls_score, self.border_bbox_pred
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    torch.nn.init.constant_(layer.weight, 1)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        torch.nn.init.constant_(self.border_cls_score.bias, bias_value)

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])

    def forward(self, features, shifts):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the K object classes.
            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every shift. These values are the
                relative offset between the shift and the ground truth box.
            centerness (list[Tensor]): #lvl tensors, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness at each spatial position.
            border_logits (list[Tensor]): #lvl tensors, each has shape (N, K, Hi, Wi).
                The tensor predicts the border classification probability
                at each spatial position for each of the K object classes.
            border_bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (dl,dt,dr,db) box
                regression values for every border shift. These values are the
                relative offset between the shift and the ground truth box.
            pre_bbox (list[Tensor]): #lvl tensors, each has shape (N, Hi * Wi, 4).
                The tensor predicts 4-vector (l,t,r,b) box regression values.
                These values are predicted boxes by the dense object detector.
        """
        logits = []
        bbox_reg = []
        centerness = []
        border_logits = []
        border_bbox_reg = []
        pre_bbox = []

        shifts = [
            torch.cat([shi.unsqueeze(0) for shi in shift], dim=0)
            for shift in list(zip(*shifts))
        ]

        for level, (feature, shifts_i) in enumerate(zip(features, shifts)):
            cls_subnet = self.cls_subnet(feature)
            bbox_subnet = self.bbox_subnet(feature)

            logits.append(self.cls_score(cls_subnet))
            if self.centerness_on_reg:
                centerness.append(self.centerness(bbox_subnet))
            else:
                centerness.append(self.centerness(cls_subnet))

            bbox_pred = self.scales[level](self.bbox_pred(bbox_subnet))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred) * self.fpn_strides[level]
            else:
                bbox_pred = torch.exp(bbox_pred) * self.fpn_strides[level]
            bbox_reg.append(bbox_pred)

            # border
            N, C, H, W = feature.shape
            pre_off = bbox_pred.clone().detach()
            with torch.no_grad():
                pre_off = pre_off.permute(0, 2, 3, 1).reshape(N, -1, 4)
                pre_boxes = self.compute_bbox(shifts_i, pre_off)
                align_boxes, wh = self.compute_border(pre_boxes, level, H, W)
                pre_bbox.append(pre_boxes)

            border_cls_conv = self.border_cls_subnet(cls_subnet, align_boxes, wh)
            border_cls_logits = self.border_cls_score(border_cls_conv)
            border_logits.append(border_cls_logits)

            border_reg_conv = self.border_bbox_subnet(bbox_subnet, align_boxes, wh)
            border_bbox_pred = self.border_bbox_pred(border_reg_conv)
            border_bbox_reg.append(border_bbox_pred)

        if self.training:
            pre_bbox = torch.cat(pre_bbox, dim=1)
        return (logits, bbox_reg, centerness, border_logits, border_bbox_reg, pre_bbox)

    def compute_bbox(self, location, pred_offset):
        detections = torch.stack([
            location[:, :, 0] - pred_offset[:, :, 0],
            location[:, :, 1] - pred_offset[:, :, 1],
            location[:, :, 0] + pred_offset[:, :, 2],
            location[:, :, 1] + pred_offset[:, :, 3]], dim=2)

        return detections

    def compute_border(self, _boxes, fm_i, height, width):
        """
        :param _boxes:
        :param fm_i:
        :param height:
        :param width:
        :return:
        """
        boxes = _boxes / self.fpn_strides[fm_i]
        boxes[:, :, 0].clamp_(min=0, max=width - 1)
        boxes[:, :, 1].clamp_(min=0, max=height - 1)
        boxes[:, :, 2].clamp_(min=0, max=width - 1)
        boxes[:, :, 3].clamp_(min=0, max=height - 1)

        wh = (boxes[:, :, 2:] - boxes[:, :, :2]).contiguous()
        return boxes, wh


class BorderBranch(nn.Module):
    def __init__(self, in_channels, border_channels):
        """
        :param in_channels:
        """
        super(BorderBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels),
            nn.ReLU())

        self.ltrb_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                border_channels * 4,
                kernel_size=1),
            nn.InstanceNorm2d(border_channels * 4),
            nn.ReLU())

        self.border_align = BorderAlign(pool_size=10)

        self.border_conv = nn.Sequential(
            nn.Conv2d(
                5 * border_channels,
                in_channels,
                kernel_size=1),
            nn.ReLU())

    def forward(self, feature, boxes):
        N, C, H, W = feature.shape

        fm_short = self.cur_point_conv(feature)
        feature = self.ltrb_conv(feature)
        ltrb_conv = self.border_align(feature, boxes)
        ltrb_conv = ltrb_conv.permute(0, 3, 1, 2).reshape(N, -1, H, W)
        align_conv = torch.cat([ltrb_conv, fm_short], dim=1)
        align_conv = self.border_conv(align_conv)
        return align_conv
