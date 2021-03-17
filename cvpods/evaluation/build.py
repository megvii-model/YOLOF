#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved
import os

import torch

from cvpods.utils import comm

from .evaluator import DatasetEvaluators
from .registry import EVALUATOR


def build_evaluator(cfg, dataset_name, dataset, output_folder=None, dump=False):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator_list = []
    meta = dataset.meta
    evaluator_type = meta.evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            EVALUATOR.get("SemSegEvaluator")(
                dataset_name,
                dataset,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
                dump=dump,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg", "citypersons"]:
        evaluator_list.append(
            EVALUATOR.get("COCOEvaluator")(dataset_name, meta, cfg, True, output_folder, dump)
        )

    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(
            EVALUATOR.get("COCOPanopticEvaluator")(dataset_name, meta, output_folder, dump))
    elif evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return EVALUATOR.get("CityscapesEvaluator")(dataset_name, meta, dump)
    elif evaluator_type == "pascal_voc":
        return EVALUATOR.get("PascalVOCDetectionEvaluator")(dataset_name, meta, dump)
    elif evaluator_type == "lvis":
        return EVALUATOR.get("LVISEvaluator")(dataset_name, meta, cfg, True, output_folder, dump)
    elif evaluator_type == "citypersons":
        evaluator_list.append(
            EVALUATOR.get("CityPersonsEvaluator")(
                dataset_name, meta, cfg, True, output_folder, dump)
        )

    if evaluator_type == "crowdhuman":
        return EVALUATOR.get("CrowdHumanEvaluator")(
            dataset_name, meta, cfg, True, output_folder, dump
        )
    elif evaluator_type == "widerface":
        return EVALUATOR.get("WiderFaceEvaluator")(
            dataset_name, meta, cfg, True, output_folder, dump)

    if evaluator_type == "classification":
        return EVALUATOR.get("ClassificationEvaluator")(
            dataset_name, meta, cfg, True, output_folder, dump)

    if hasattr(cfg, "EVALUATORS"):
        for evaluator in cfg.EVALUATORS:
            evaluator_list.append(evaluator(dataset_name, meta, True, output_folder, dump=True))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)
