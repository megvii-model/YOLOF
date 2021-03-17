#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import copy
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util

import torch

from cvpods.data.datasets.coco import convert_to_coco_json
from cvpods.structures import BoxMode
from cvpods.utils import PathManager, comm, create_small_table

from .crowdhumantools import Database
from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class CrowdHumanEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """

    def __init__(
            self,
            dataset_name,
            meta,
            cfg,
            distributed,
            output_dir=None,
            dump=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in cvpods's standard dataset format
                so it can be converted to COCO format automatically.
            meta (dict): dataset meta.
            cfg (dict): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            dump (boolean): optional, whether dump predictions to disk.
        """
        self._dump = dump
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = meta
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        # TODO@wangfeng02: next 4 line to a func
        if self._dump:
            with open("README.md", "w") as f:
                name = cfg.OUTPUT_DIR.split("/")[-1]
                f.write("# {}  \n".format(name))

        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def boxes_dump(self, boxes, is_gt=False):
        result = []
        boxes = boxes.tolist()
        for box in boxes:
            if is_gt:
                box_dict = {}
                box_dict['box'] = [box[0], box[1], box[2] - box[0],
                                   box[3] - box[1]]
                box_dict['tag'] = box[-1]
                result.append(box_dict)
            else:
                box_dict = {}
                box_dict['box'] = [box[0], box[1], box[2] - box[0],
                                   box[3] - box[1]]
                box_dict['tag'] = 1
                box_dict['score'] = box[-1]
                result.append(box_dict)
        return result

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(
                    self._cpu_device)

            gt_boxes = input['instances'].gt_boxes.tensor.cpu().numpy()
            gt_classes = input['instances'].gt_classes.cpu().numpy()[:, np.newaxis]
            gt_boxes = np.concatenate([gt_boxes, gt_classes], axis=1)

            pred_boxes = output['instances'].pred_boxes.tensor.cpu().numpy()
            pred_score = output['instances'].scores.cpu().numpy()[:, np.newaxis]
            pred_boxes = np.concatenate([pred_boxes, pred_score], axis=1)

            result_dict = dict(
                ID=input['image_id'],
                height=int(input['height']),
                width=int(input['width']),
                dtboxes=self.boxes_dump(pred_boxes),
                gtboxes=self.boxes_dump(gt_boxes, is_gt=True)
            )
            # rois=misc_utils.boxes_dump(rois[:, 1:], True))
            self._predictions.append(result_dict)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        self._eval_predictions(set(self._tasks))
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for CrowdHuman format ...")
        self._coco_results = self._predictions

        if self._output_dir:
            file_path = os.path.join(
                self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))

            with PathManager.open(file_path, "w") as f:
                for db in self._coco_results:
                    line = json.dumps(db) + '\n'
                    f.write(line)

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_crowdhuman(
                    self._metadata.json_file, file_path)
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(coco_eval, task)
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "mMR", "Recall"]

        if coco_eval is None:
            self._logger.warn(
                "No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {metric: coco_eval[idx]
                   for idx, metric in enumerate(metrics)}
        small_table = create_small_table(results)
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + small_table
        )

        # if class_names is None or len(class_names) <= 1:
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.
    Args:
        instances (Instances):
        img_id (int): the image id
    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "height": boxes[k][3],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def _evaluate_predictions_on_crowdhuman(gt_path, dt_path, target_key="box", mode=0):
    """
    Evaluate the coco results using COCOEval API.
    """
    database = Database(gt_path, dt_path, target_key, None, mode)
    database.compare()
    AP, recall, _ = database.eval_AP()
    mMR, _ = database.eval_MR()
    return AP, mMR, recall
