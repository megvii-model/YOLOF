# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by BaseDetection, Inc. and its affiliates. All Rights Reserved

# pylint: disable=W0613

"""
Detection Training Script.
This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in cvpods.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use cvpods as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import logging
import os
from collections import OrderedDict
from colorama import Fore, Style

from cvpods.engine import RUNNERS, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import build_evaluator
from cvpods.modeling import GeneralizedRCNNWithTTA

from config import config
from net import build_model


def runner_decrator(cls):
    """
    We use the "DefaultRunner" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleRunner", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    def custom_build_evaluator(cls, cfg, dataset_name, dataset, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        dump_train = config.GLOBAL.DUMP_TRAIN
        return build_evaluator(cfg, dataset_name, dataset, output_folder, dump=dump_train)

    def custom_test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("cvpods.runner")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        res = cls.test(cfg, model, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    cls.build_evaluator = classmethod(custom_build_evaluator)
    cls.test_with_TTA = classmethod(custom_test_with_TTA)

    return cls


def main(args):
    config.merge_from_list(args.opts)
    cfg, logger = default_setup(config, args)

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the runner.
    """
    runner = runner_decrator(RUNNERS.get(cfg.TRAINER.NAME))(cfg, build_model)
    runner.resume_or_load(resume=args.resume)

    # check wheather worksapce has enough storeage space
    # assume that a single dumped model is 700Mb
    file_sys = os.statvfs(cfg.OUTPUT_DIR)
    free_space_Gb = (file_sys.f_bfree * file_sys.f_frsize) / 2**30
    eval_space_Gb = (cfg.SOLVER.LR_SCHEDULER.MAX_ITER // cfg.SOLVER.CHECKPOINT_PERIOD) * 700 / 2**10
    if eval_space_Gb > free_space_Gb:
        logger.warning(f"{Fore.RED}Remaining space({free_space_Gb}GB) "
                       f"is less than ({eval_space_Gb}GB){Style.RESET_ALL}")

    if cfg.TEST.AUG.ENABLED:
        runner.register_hooks(
            [hooks.EvalHook(0, lambda: runner.test_with_TTA(cfg, runner.model))]
        )

    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info("different config with base class:\n{}".format(cfg.diff(base_config)))

    runner.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    config.link_log()
    print("soft link to {}".format(config.OUTPUT_DIR))
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
