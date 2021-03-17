#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved
"""
Debuging Script.
"""
import logging
import os
import re

from cvpods.checkpoint import Checkpointer
from cvpods.engine import TrainerBase, default_argument_parser, default_setup, launch
from cvpods.solver import build_optimizer
from cvpods.utils import comm

from config import config
from net import build_model


class DebugTrainer(TrainerBase):

    def __init__(self, model, data, optimizer):
        model.train()

        self.data = data
        self.model = model
        self.optimizer = optimizer

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"

        self.optimizer.zero_grad()

        # for each mini step
        loss_dict = self.model(self.data)

        for metrics_name, metrics_value in loss_dict.items():
            # Actually, some metrics are not loss, such as
            # top1_acc, top5_acc in classification, filter them out
            if metrics_value.requires_grad:
                loss_dict[metrics_name] = metrics_value

        losses = sum([
            metrics_value for metrics_value in loss_dict.values()
            if metrics_value.requires_grad
        ])
        losses.backward()

        self.optimizer.step()


def debug_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--ckpt-file", type=str, default=None, help="path of debug checkpoint file"
    )
    return parser


def stage_main(args, cfg, build):
    logger = logging.getLogger(__name__)
    assert comm.get_world_size() == 1, "DEBUG mode only supported for 1 GPU"

    cfg.merge_from_list(args.opts)
    cfg, logger = default_setup(cfg, args)
    model = build(cfg)
    optimizer = build_optimizer(cfg, model)
    debug_ckpt = Checkpointer(model, resume=True, optimizer=optimizer)
    ckpt_file = args.ckpt_file
    if ckpt_file is None:
        # find latest checkpoint in log dir if ckpt_file is not given
        log_dir = "./log"
        matched_files = [
            os.path.join(log_dir, files) for files in os.listdir(log_dir)
            if re.match("debug_.*.pth", files) is not None
        ]
        ckpt_file = sorted(matched_files, key=os.path.getatime)[-1]

    left_dict = debug_ckpt.load(ckpt_file)
    assert "inputs" in left_dict, "input data not found in checkpoints"
    data = left_dict["inputs"]

    trainer = DebugTrainer(model, data, optimizer)
    logger.info("start run models")
    trainer.run_step()
    logger.info("finish debuging")


def main(args):
    if isinstance(config, list):
        assert isinstance(build_model, list) and len(config) == len(build_model)
        for cfg, build in zip(config, build_model):
            stage_main(args, cfg, build)
    else:
        stage_main(args, config, build_model)


if __name__ == "__main__":
    args = debug_parser().parse_args()

    if isinstance(config, list):
        assert len(config) > 0
        print("soft link first config in list to {}".format(config[0].OUTPUT_DIR))
        config[0].link_log()
    else:
        print("soft link to {}".format(config.OUTPUT_DIR))
        config.link_log()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
