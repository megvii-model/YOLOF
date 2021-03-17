# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import glob
import logging
import os
import tempfile
from collections import OrderedDict
from tabulate import tabulate

from PIL import Image

import torch

from cvpods.utils import comm, create_small_table

from .evaluator import DatasetEvaluator
from .registry import EVALUATOR


@EVALUATOR.register()
class CityscapesEvaluator(DatasetEvaluator):
    """
    Evaluate instance segmentation results using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
    """

    def __init__(self, dataset_name, meta, dump=False):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
            meta (SimpleNamespace): dataset metadata.
            dump (bool): If True, after the evaluation is completed, a Markdown file
                that records the model evaluation metrics and corresponding scores
                will be generated in the working directory.
        """
        # TODO: really use dataset_name
        self.dataset_name = dataset_name
        self._dump = dump
        self._metadata = meta
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_")
        self._temp_dir = self._working_dir.name
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )

    def process(self, inputs, outputs):
        from cityscapesscripts.helpers.labels import name2label

        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_txt = os.path.join(self._temp_dir, basename + "_pred.txt")

            output = output["instances"].to(self._cpu_device)
            num_instances = len(output)
            with open(pred_txt, "w") as fout:
                for i in range(num_instances):
                    pred_class = output.pred_classes[i]
                    classes = self._metadata.thing_classes[pred_class]
                    class_id = name2label[classes].id
                    score = output.scores[i]
                    mask = output.pred_masks[i].numpy().astype("uint8")
                    png_filename = os.path.join(
                        self._temp_dir, basename + "_{}_{}.png".format(i, classes)
                    )

                    Image.fromarray(mask * 255).save(png_filename)
                    fout.write("{} {} {}\n".format(os.path.basename(png_filename), class_id, score))

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        os.environ["CITYSCAPES_DATASET"] = os.path.abspath(
            os.path.join(self._metadata.gt_dir, "..", "..")
        )
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        import cityscapesscripts.evaluation.evalInstanceLevelSemanticLabeling as cityscapes_eval

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False
        cityscapes_eval.args.gtInstancesFile = os.path.join(self._temp_dir, "gtInstances.json")

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa
        groundTruthImgList = glob.glob(cityscapes_eval.args.groundTruthSearch)
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(cityscapes_eval.getPrediction(gt, cityscapes_eval.args))
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )["averages"]

        ret = OrderedDict()
        ret["segm"] = {"AP": results["allAp"] * 100, "AP50": results["allAp50%"] * 100}
        self._working_dir.cleanup()

        small_table = create_small_table(ret["segm"])
        self._logger.info("Evaluation results for segm: \n" + small_table)

        results_per_category = []
        for cat, ap in results["classes"].items():
            ap = [ap_i * 100 for ap_i in ap.values()]
            results_per_category.append([cat, *ap])

        table = tabulate(
            results_per_category,
            headers=["category", "AP", "AP50"],
            tablefmt="pipe",
            floatfmt=".3f",
            numalign="left"
        )
        self._logger.info("Per-category segm AP: \n" + table)

        if self._dump:
            dump_info_one_task = {
                "task": "segm",
                "tables": [small_table, table],
            }
            _dump_to_markdown([dump_info_one_task])
        return ret


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
            f.write("\n\n### Per-category {} AP:  \n\n".format(task_name))
            f.write(tables[1])
            f.write("\n")
