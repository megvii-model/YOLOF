#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import copy
import itertools
import logging
import os
from collections import OrderedDict

import torch

from cvpods.utils import PathManager, comm, create_small_table

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class ClassificationEvaluator(DatasetEvaluator):
    """
    Evaluate instance calssification results.
    """

    # TODO: unused_arguments: cfg
    def __init__(self, dataset_name, meta, cfg, distributed, output_dir=None, dump=False):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            meta (SimpleNamespace): dataset metadata.
            cfg (config dict): cvpods Config instance.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.

            dump (bool): If True, after the evaluation is completed, a Markdown file
                that records the model evaluation metrics and corresponding scores
                will be generated in the working directory.
        """
        super(ClassificationEvaluator, self).__init__()
        # TODO: really use dataset_name
        self.dataset_name = dataset_name
        self._dump = dump
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = meta

        self._topk = (1, 5)

    def reset(self):
        self._predictions = []
        self._targets = []
        self._dump_infos = []  # per task

    def process(self, inputs, outputs):
        # Find the top max_k predictions for each sample
        _top_max_k_vals, top_max_k_inds = torch.topk(
            outputs.cpu(), max(self._topk), dim=1, largest=True, sorted=True
        )
        # (batch_size, max_k) -> (max_k, batch_size)
        top_max_k_inds = top_max_k_inds.t()

        self._targets.append(torch.tensor([i["category_id"] for i in inputs]))
        self._predictions.append(top_max_k_inds)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            self._targets = comm.gather(self._targets, dst=0)
            self._targets = list(itertools.chain(*self._targets))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[ClassificationEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        assert len(self._predictions) == len(self._targets)
        if self._predictions[0] is not None:
            self._eval_classification_accuracy()

        if self._dump:
            _dump_to_markdown(self._dump_infos)

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_classification_accuracy(self):
        """
        Evaluate self._predictions on the classification task.
        Fill self._results with the metrics of the tasks.
        """
        batch_size = len(self._targets)

        pred = torch.cat(self._predictions, dim=1)
        target = torch.cat(self._targets)

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = {}
        for k in self._topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            results[f"Top_{k} Acc"] = correct_k.mul_(100.0 / batch_size).item()
        self._results["Accuracy"] = results

        small_table = create_small_table(results)
        self._logger.info("Evaluation results for classification: \n" + small_table)

        if self._dump:
            dump_info_one_task = {
                "task": "classification",
                "tables": [small_table],
            }
            self._dump_infos.append(dump_info_one_task)


def _dump_to_markdown(dump_infos, md_file="README.md"):
    """
    Dump a Markdown file that records the model evaluation metrics and corresponding scores
    to the current working directory.

    Args:
        dump_infos (list[dict]): dump information for each task.
        md_file (str): markdown file path.
    """
    with open(md_file, "w") as f:
        title = os.path.basename(os.getcwd())
        f.write("# {}  ".format(title))
        for dump_info_per_task in dump_infos:
            task_name = dump_info_per_task["task"]
            tables = dump_info_per_task["tables"]
            tables = [table.replace("\n", "  \n") for table in tables]
            f.write("\n\n## Evaluation results for {}:  \n\n".format(task_name))
            f.write(tables[0] + "\n")
