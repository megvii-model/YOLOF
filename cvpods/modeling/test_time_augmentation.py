# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from contextlib import contextmanager
from itertools import count

import numpy as np

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from cvpods.data.detection_utils import read_image
from cvpods.data.transforms import ResizeShortestEdge
from cvpods.layers import generalized_batched_nms
from cvpods.structures import Boxes, Instances

from .meta_arch import GeneralizedRCNN
from .postprocessing import detector_postprocess
from .roi_heads.fast_rcnn import fast_rcnn_inference_single_image

__all__ = ["DatasetMapperTTA", "GeneralizedRCNNWithTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT
        self.extra_sizes = cfg.TEST.AUG.EXTRA_SIZES

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a detection dataset dict

        Returns:
            list[dict]:
                a list of dataset dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
        """
        ret = []
        if "image" not in dataset_dict:
            numpy_image = read_image(dataset_dict["file_name"], self.image_format)
        else:
            numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy().astype("uint8")

        image_sizes = [(min_size, self.max_size) for min_size in self.min_sizes]
        image_sizes.extend(self.extra_sizes)

        for min_size, max_size in image_sizes:
            image = np.copy(numpy_image)
            tfm = ResizeShortestEdge(min_size, max_size).get_transform(image)
            resized = tfm.apply_image(image)
            resized = torch.as_tensor(resized.transpose(2, 0, 1).astype("float32"))

            dic = copy.deepcopy(dataset_dict)
            dic["horiz_flip"] = False
            dic["image"] = resized
            ret.append(dic)

            if self.flip:
                dic = copy.deepcopy(dataset_dict)
                dic["horiz_flip"] = True
                dic["image"] = torch.flip(resized, dims=[2])
                ret.append(dic)

        return ret


class GeneralizedRCNNWithTTA(nn.Module):
    """
    A GeneralizedRCNN with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (config dict):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, GeneralizedRCNN
        ), "TTA is only supported on GeneralizedRCNN. Got a model of type {}".format(type(model))
        self.cfg = copy.deepcopy(cfg)
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    @contextmanager
    def _turn_off_roi_head(self, attr):
        """
        Open a context where one head in `model.roi_heads` is temporarily turned off.
        Args:
            attr (str): the attribute in `model.roi_heads` which can be used
                to turn off a specific head, e.g., "mask_on", "keypoint_on".
        """
        roi_heads = self.model.roi_heads
        try:
            old = getattr(roi_heads, attr)
        except AttributeError:
            # The head may not be implemented in certain ROIHeads
            old = None

        if old is None:
            yield
        else:
            setattr(roi_heads, attr, False)
            yield
            setattr(roi_heads, attr, old)

    def _batch_inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=do_postprocess,
                    )
                )
                inputs, instances = [], []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """
        return [self._inference_one_image(x) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict

        Returns:
            dict: one output dict
        """
        augmented_inputs = self.tta_mapper(input)

        do_hflip = [k.pop("horiz_flip", False) for k in augmented_inputs]
        heights = [k["height"] for k in augmented_inputs]
        widths = [k["width"] for k in augmented_inputs]
        assert (
            len(set(heights)) == 1 and len(set(widths)) == 1
        ), "Augmented version of the inputs should have the same original resolution!"
        height = heights[0]
        width = widths[0]

        # 1. Detect boxes from all augmented versions
        # 1.1: forward with all augmented images
        with self._turn_off_roi_head("mask_on"), self._turn_off_roi_head("keypoint_on"):
            # temporarily disable mask/keypoint head
            outputs = self._batch_inference(augmented_inputs, do_postprocess=False)
        # 1.2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for idx, output in enumerate(outputs):
            rescaled_output = detector_postprocess(output, height, width)
            pred_boxes = rescaled_output.pred_boxes.tensor
            if do_hflip[idx]:
                pred_boxes[:, [0, 2]] = width - pred_boxes[:, [2, 0]]
            all_boxes.append(pred_boxes)
            all_scores.extend(rescaled_output.scores)
            all_classes.extend(rescaled_output.pred_classes)
        all_boxes = torch.cat(all_boxes, dim=0).cpu()
        num_boxes = len(all_boxes)

        # 1.3: select from the union of all results
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            (height, width),
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.MODEL.NMS_TYPE,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        if not self.cfg.MODEL.MASK_ON:
            return {"instances": merged_instances}

        # 2. Use the detected boxes to obtain masks
        # 2.1: rescale the detected boxes
        augmented_instances = []
        for idx, input in enumerate(augmented_inputs):
            actual_height, actual_width = input["image"].shape[1:3]
            scale_x = actual_width * 1.0 / width
            scale_y = actual_height * 1.0 / height
            pred_boxes = merged_instances.pred_boxes.clone()
            pred_boxes.tensor[:, 0::2] *= scale_x
            pred_boxes.tensor[:, 1::2] *= scale_y
            if do_hflip[idx]:
                pred_boxes.tensor[:, [0, 2]] = actual_width - pred_boxes.tensor[:, [2, 0]]

            aug_instances = Instances(
                image_size=(actual_height, actual_width),
                pred_boxes=pred_boxes,
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        # 2.2: run forward on the detected boxes
        outputs = self._batch_inference(augmented_inputs, augmented_instances, do_postprocess=False)
        for idx, output in enumerate(outputs):
            if do_hflip[idx]:
                output.pred_masks = output.pred_masks.flip(dims=[3])
        # 2.3: average the predictions
        all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        output = outputs[0]
        output.pred_masks = avg_pred_masks
        output = detector_postprocess(output, height, width)
        return {"instances": output}


class SimpleTTAWarper(nn.Module):

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (config dict):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """
        return [self._inference_one_image(x) for x in batched_inputs]

    def _inference_one_image(self, inputs):
        augmented_inputs = self.tta_mapper(inputs)
        assert len({x["file_name"] for x in augmented_inputs}) == 1, "inference different images"
        heights = [k["height"] for k in augmented_inputs]
        widths = [k["width"] for k in augmented_inputs]
        assert (
            len(set(heights)) == 1
            and len(set(widths)) == 1
        ), "Augmented version of the inputs should have the same original resolution!"

        height = heights[0]
        width = widths[0]
        # 1. Detect boxes from all augmented versions
        all_boxes = []
        all_scores = []
        all_classes = []

        for single_input in augmented_inputs:
            do_hflip = single_input.pop("horiz_flip", False)
            # 1.1: forward with single augmented image
            output = self.model._inference_for_ms_test([single_input])
            # 1.2: union the results
            pred_boxes = output.get("pred_boxes").tensor
            if do_hflip:
                pred_boxes[:, [0, 2]] = width - pred_boxes[:, [2, 0]]
            all_boxes.append(pred_boxes)
            all_scores.append(output.get("scores"))
            all_classes.append(output.get("pred_classes"))

        boxes_all = torch.cat(all_boxes, dim=0)
        scores_all = torch.cat(all_scores, dim=0)
        class_idxs_all = torch.cat(all_classes, dim=0)
        keep = generalized_batched_nms(
            boxes_all, scores_all, class_idxs_all,
            self.model.nms_threshold, nms_type=self.model.nms_type
        )

        keep = keep[:self.model.max_detections_per_image]

        result = Instances((height, width))
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return {"instances": result}


class TTAWarper(nn.Module):

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (config dict):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size
        self.max_detection = cfg.TEST.DETECTIONS_PER_IMAGE

        self.enable_scale_filter = cfg.TEST.AUG.SCALE_FILTER
        self.scale_ranges = cfg.TEST.AUG.SCALE_RANGES

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """
        return [self._inference_one_image(x) for x in batched_inputs]

    def _inference_one_image(self, inputs):
        augmented_inputs = self.tta_mapper(inputs)
        assert len({x["file_name"] for x in augmented_inputs}) == 1, "inference different images"
        heights = [k["height"] for k in augmented_inputs]
        widths = [k["width"] for k in augmented_inputs]
        assert (
            len(set(heights)) == 1
            and len(set(widths)) == 1
        ), "Augmented version of the inputs should have the same original resolution!"

        height = heights[0]
        width = widths[0]
        # 1. Detect boxes from all augmented versions
        # TODO wangfeng02: use box structures instead of boxes, scores and classes
        all_boxes = []
        all_scores = []
        all_classes = []

        factors = 2 if self.tta_mapper.flip else 1
        if self.enable_scale_filter:
            assert len(augmented_inputs) == len(self.scale_ranges) * factors

        for i, single_input in enumerate(augmented_inputs):
            do_hflip = single_input.pop("horiz_flip", False)
            # 1.1: forward with single augmented image
            output = self.model._inference_for_ms_test([single_input])
            # 1.2: union the results
            pred_boxes = output.get("pred_boxes").tensor
            if do_hflip:
                pred_boxes[:, [0, 2]] = width - pred_boxes[:, [2, 0]]

            pred_scores = output.get("scores")
            pred_classes = output.get("pred_classes")
            if self.enable_scale_filter:
                keep = filter_boxes(pred_boxes, *self.scale_ranges[i // factors])
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_classes = pred_classes[keep]

            all_boxes.append(pred_boxes)
            all_scores.append(pred_scores)
            all_classes.append(pred_classes)

        boxes_all = torch.cat(all_boxes, dim=0)
        scores_all = torch.cat(all_scores, dim=0)
        class_idxs_all = torch.cat(all_classes, dim=0)
        boxes_all, scores_all, class_idxs_all = merge_result_from_multi_scales(
            boxes_all, scores_all, class_idxs_all,
            nms_type="soft_vote", vote_thresh=0.65,
            max_detection=self.max_detection
        )

        result = Instances((height, width))
        result.pred_boxes = Boxes(boxes_all)
        result.scores = scores_all
        result.pred_classes = class_idxs_all
        return {"instances": result}


def filter_boxes(boxes, min_scale, max_scale):
    """
    boxes: (N, 4) shape
    """
    # assert boxes.mode == "xyxy"
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return keep


def merge_result_from_multi_scales(
    boxes, scores, labels, nms_type="soft-vote", vote_thresh=0.65, max_detection=100
):
    boxes, scores, labels = batched_vote_nms(
        boxes, scores, labels, nms_type, vote_thresh
    )

    number_of_detections = boxes.shape[0]
    # Limit to max_per_image detections **over all classes**
    if number_of_detections > max_detection > 0:
        boxes = boxes[:max_detection]
        scores = scores[:max_detection]
        labels = labels[:max_detection]

    return boxes, scores, labels


def batched_vote_nms(boxes, scores, labels, vote_type, vote_thresh=0.65):
    # apply per class level nms, add max_coordinates on boxes first, then remove it.
    labels = labels.float()
    max_coordinates = boxes.max() + 1
    offsets = labels.reshape(-1, 1) * max_coordinates
    boxes = boxes + offsets

    boxes, scores, labels = bbox_vote(boxes, scores, labels, vote_thresh, vote_type)
    boxes -= labels.reshape(-1, 1) * max_coordinates

    return boxes, scores, labels


def bbox_vote(boxes, scores, labels, vote_thresh, vote_type="softvote"):
    assert boxes.shape[0] == scores.shape[0] == labels.shape[0]
    det = torch.cat((boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)), dim=1)

    vote_results = torch.zeros(0, 6, device=det.device)
    if det.numel() == 0:
        return vote_results[:, :4], vote_results[:, 4], vote_results[:, 5]

    order = scores.argsort(descending=True)
    det = det[order]

    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        xx1 = torch.max(det[0, 0], det[:, 0])
        yy1 = torch.max(det[0, 1], det[:, 1])
        xx2 = torch.min(det[0, 2], det[:, 2])
        yy2 = torch.min(det[0, 3], det[:, 3])
        w = torch.clamp(xx2 - xx1, min=0.)
        h = torch.clamp(yy2 - yy1, min=0.)
        inter = w * h
        iou = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = torch.where(iou >= vote_thresh)[0]
        vote_det = det[merge_index, :]
        det = det[iou < vote_thresh]

        if merge_index.shape[0] <= 1:
            vote_results = torch.cat((vote_results, vote_det), dim=0)
        else:
            if vote_type == "soft_vote":
                vote_det_iou = iou[merge_index]
                det_accu_sum = get_soft_dets_sum(vote_det, vote_det_iou)
            elif vote_type == "vote":
                det_accu_sum = get_dets_sum(vote_det)
            vote_results = torch.cat((vote_results, det_accu_sum), dim=0)

    order = vote_results[:, 4].argsort(descending=True)
    vote_results = vote_results[order, :]

    return vote_results[:, :4], vote_results[:, 4], vote_results[:, 5]


def get_dets_sum(vote_det):
    vote_det[:, :4] *= vote_det[:, 4:5].repeat(1, 4)
    max_score = vote_det[:, 4].max()
    det_accu_sum = torch.zeros((1, 6), device=vote_det.device)
    det_accu_sum[:, :4] = torch.sum(vote_det[:, :4], dim=0) / torch.sum(vote_det[:, 4])
    det_accu_sum[:, 4] = max_score
    det_accu_sum[:, 5] = vote_det[0, 5]
    return det_accu_sum


def get_soft_dets_sum(vote_det, vote_det_iou):
    soft_vote_det = vote_det.detach().clone()
    soft_vote_det[:, 4] *= (1 - vote_det_iou)

    INFERENCE_TH = 0.05
    soft_index = torch.where(soft_vote_det[:, 4] >= INFERENCE_TH)[0]
    soft_vote_det = soft_vote_det[soft_index, :]

    vote_det[:, :4] *= vote_det[:, 4:5].repeat(1, 4)
    max_score = vote_det[:, 4].max()
    det_accu_sum = torch.zeros((1, 6), device=vote_det.device)
    det_accu_sum[:, :4] = torch.sum(vote_det[:, :4], dim=0) / torch.sum(vote_det[:, 4])
    det_accu_sum[:, 4] = max_score
    det_accu_sum[:, 5] = vote_det[0, 5]

    if soft_vote_det.shape[0] > 0:
        det_accu_sum = torch.cat((det_accu_sum, soft_vote_det), dim=0)
    return det_accu_sum
