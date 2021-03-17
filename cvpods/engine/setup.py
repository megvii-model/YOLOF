# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import argparse
import os

import torch

from cvpods.utils import PathManager, collect_env_info, comm, seed_all_rng, setup_logger

__all__ = ["default_argument_parser", "default_setup"]


def default_argument_parser():
    """
    Create a parser with some common arguments used by cvpods users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="cvpods Training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only",
                        action="store_true",
                        help="perform evaluation only")
    parser.add_argument("--num-gpus",
                        type=int,
                        default=1,
                        help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank",
                        type=int,
                        default=0,
                        help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid()) % 2**14
    parser.add_argument("--dist-url",
                        default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def check_subdivision_config(cfg):
    images_per_device = cfg.SOLVER.IMS_PER_DEVICE
    batch_subdivisions = cfg.SOLVER.BATCH_SUBDIVISIONS

    assert (
        batch_subdivisions > 0
    ), "SOLVER.BATCH_SUBDIVISIONS ({}) must be a positive number.".format(
        batch_subdivisions
    )

    if batch_subdivisions > 1:
        # if batch_subdivisions is equal to 1, the following check is redundant
        assert (
            images_per_device % batch_subdivisions == 0
        ), "SOLVER.IMS_PER_DEVICE ({}) must be divisible by the " \
            "SOLVER.BATCH_SUBDIVISIONS ({}).".format(images_per_device, batch_subdivisions)


def adjust_config(cfg):
    base_world_size = int(cfg.SOLVER.IMS_PER_BATCH / cfg.SOLVER.IMS_PER_DEVICE)
    # Batchsize, learning rate and max_iter in original config is used for 8 GPUs
    assert base_world_size == 8, "IMS_PER_BATCH/DEVICE in config file is used for 8 GPUs"
    world_size = comm.get_world_size()
    machines_ratio = world_size / base_world_size

    # ------ adjust batch_size ---------- #
    cfg.SOLVER.IMS_PER_BATCH = int(machines_ratio * cfg.SOLVER.IMS_PER_BATCH)
    assert (
        cfg.SOLVER.IMS_PER_BATCH / cfg.SOLVER.IMS_PER_DEVICE == world_size
    ), "IMS_PER_BATCH ({}) not equal to IMS_PER_BATCH ({}) * world_size ({})".format(
        cfg.SOLVER.IMS_PER_BATCH, cfg.SOLVER.IMS_PER_DEVICE, world_size
    )
    check_subdivision_config(cfg)

    # ------- adjust scheduler --------- #
    # since we use new IMS_PER_BATCH value, epoch value doesn't need to multiply ratio
    if cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH is None:
        cfg.SOLVER.LR_SCHEDULER.MAX_ITER = int(cfg.SOLVER.LR_SCHEDULER.MAX_ITER / machines_ratio)
        cfg.SOLVER.LR_SCHEDULER.STEPS = [
            int(step / machines_ratio) for step in cfg.SOLVER.LR_SCHEDULER.STEPS
        ]
        cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.CHECKPOINT_PERIOD / machines_ratio)
        cfg.TEST.EVAL_PERIOD = int(cfg.TEST.EVAL_PERIOD / machines_ratio)

    if "SGD" in cfg.SOLVER.OPTIMIZER.NAME:
        # adjust learning rate according to Linear rule
        cfg.SOLVER.OPTIMIZER.BASE_LR = machines_ratio * cfg.SOLVER.OPTIMIZER.BASE_LR


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the cvpods logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (BaseConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    # setup_logger(output_dir, distributed_rank=rank, name="cvpods")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(
        rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info("Contents of args.config_file={}:\n{}".format(
            args.config_file,
            PathManager.open(args.config_file, "r").read())
        )

    adjust_config(cfg)
    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info("different config with base class:\n{}".format(cfg.diff(base_config)))
    # if comm.is_main_process() and output_dir:
    #     # Note: some of our scripts may expect the existence of
    #     # config.yaml in output directory
    #     path = os.path.join(output_dir, "config.yaml")
    #     with PathManager.open(path, "w") as f:
    #         f.write(cfg.dump())
    #     logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed = seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)
    # save seed to config for dump
    cfg.SEED = seed

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    return cfg, logger
