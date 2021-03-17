# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import PIL.Image as Image
import pycocotools.mask as mask_util

import torch

from cvpods.utils import PathManager, comm, create_small_table

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation results.
    """

    def __init__(self,
                 dataset_name,
                 dataset,
                 distributed,
                 num_classes,
                 ignore_label=255,
                 output_dir=None,
                 dump=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            dataset (Dataset): the dataset used for evaluation.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            num_classes (int): number of classes.
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
                corresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
            dump (bool): If True, after the evaluation is completed, a Markdown file
                that records the model evaluation metrics and corresponding scores
                will be generated in the working directory.
        """
        self._dump = dump
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        dataset_dicts = []
        if hasattr(dataset, "datasets"):
            for d in dataset.datasets:
                dataset_dicts += d.dataset_dicts
        else:
            dataset_dicts = dataset.dataset_dicts
        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in dataset_dicts
        }

        # Dict that maps contiguous training ids to COCO category ids
        meta = dataset.meta
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None

    def reset(self):
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            with PathManager.open(
                    self.input_file_to_gt_file[input["file_name"]], "rb") as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                self._N * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._N**2).reshape(self._N, self._N)

            self._predictions.extend(
                self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            comm.synchronize()
            conf_matrix_list = comm.all_gather(self._conf_matrix)
            self._predictions = comm.all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not comm.is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir,
                                     "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        if self._output_dir:
            file_path = os.path.join(self._output_dir,
                                     "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})

        small_table = create_small_table(res)
        self._logger.info("Evaluation results for sem_seg: \n" + small_table)

        if self._dump:
            dump_info_one_task = {
                "task": "sem_seg",
                "tables": [small_table],
            }
            _dump_to_markdown([dump_info_one_task])

        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(
                    label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None],
                                                 order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append({
                "file_name": input_file_name,
                "category_id": dataset_id,
                "segmentation": mask_rle
            })
        return json_list


def _dump_to_markdown(dump_infos, md_file="README.md"):
    """
    Dump a Markdown file that records the model evaluation metrics and corresponding scores
    to the current working directory.

    Args:
        dump_infos (list[dict]): dump information for each task.
        md_file (str): markdown file path.
    """
    title = os.getcwd().split("/")[-1]
    with open(md_file, "w") as f:
        f.write("# {}  ".format(title))
        for dump_info_per_task in dump_infos:
            task_name = dump_info_per_task["task"]
            tables = dump_info_per_task["tables"]
            tables = [table.replace("\n", "  \n") for table in tables]
            f.write("\n\n## Evaluation results for {}:  \n\n".format(task_name))
            f.write(tables[0])
            f.write("\n")
