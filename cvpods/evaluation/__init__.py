# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .build import build_evaluator
from .citypersons_evaluation import CityPersonsEvaluator
from .cityscapes_evaluation import CityscapesEvaluator
from .classification_evaluation import ClassificationEvaluator
from .coco_evaluation import COCOEvaluator
from .crowdhuman_evaluation import CrowdHumanEvaluator
from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_context, inference_on_dataset
from .lvis_evaluation import LVISEvaluator
from .panoptic_evaluation import COCOPanopticEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .registry import EVALUATOR
from .rotated_coco_evaluation import RotatedCOCOEvaluator
from .sem_seg_evaluation import SemSegEvaluator
from .testing import print_csv_format, verify_results
from .widerface_evaluation import WiderFaceEvaluator

__all__ = [k for k in globals().keys() if not k.startswith("_")]
