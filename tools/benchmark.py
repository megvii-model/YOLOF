# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to benchmark builtin models.

Note: this script has an extra dependency of psutil.
"""

import itertools
import logging
import psutil
import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.config import get_cfg
from cvpods.data import DatasetFromList, build_test_loader, build_train_loader
from cvpods.engine import SimpleTrainer, default_argument_parser, hooks, launch
from cvpods.modeling import build_model
from cvpods.solver import build_optimizer
from cvpods.utils import CommonMetricPrinter, Timer, comm, setup_logger

logger = logging.getLogger("cvpods")


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.SOLVER.BASE_LR = 0.001  # Avoid NaNs. Not useful in this script anyway.
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    setup_logger(distributed_rank=comm.get_rank())
    return cfg


def benchmark_data(args):
    cfg = setup(args)

    dataloader = build_train_loader(cfg)

    timer = Timer()
    itr = iter(dataloader)
    for i in range(10):  # warmup
        next(itr)
        if i == 0:
            startup_time = timer.seconds()
    timer = Timer()
    max_iter = 1000
    for _ in tqdm.trange(max_iter):
        next(itr)
    logger.info(
        "{} iters ({} images) in {} seconds.".format(
            max_iter, max_iter * cfg.SOLVER.IMS_PER_BATCH, timer.seconds()
        )
    )
    logger.info("Startup time: {} seconds".format(startup_time))
    vram = psutil.virtual_memory()
    logger.info(
        "RAM Usage: {:.2f}/{:.2f} GB".format(
            (vram.total - vram.available) / 1024 ** 3, vram.total / 1024 ** 3
        )
    )


def benchmark_train(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    optimizer = build_optimizer(cfg, model)
    checkpointer = DefaultCheckpointer(model, optimizer=optimizer)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    data_loader = build_train_loader(cfg)
    dummy_data = list(itertools.islice(data_loader, 100))

    def f():
        while True:
            yield from DatasetFromList(dummy_data, copy=False)

    max_iter = 400
    trainer = SimpleTrainer(model, f(), optimizer)
    trainer.register_hooks(
        [hooks.IterationTimer(), hooks.PeriodicWriter([CommonMetricPrinter(max_iter)])]
    )
    trainer.train(1, max_iter)


@torch.no_grad()
def benchmark_eval(args):
    cfg = setup(args)
    model = build_model(cfg)
    model.eval()
    logger.info("Model:\n{}".format(model))
    DefaultCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    data_loader = build_test_loader(cfg, cfg.DATASETS.TEST[0])
    dummy_data = list(itertools.islice(data_loader, 100))

    def f():
        while True:
            yield from DatasetFromList(dummy_data, copy=False)

    for _ in range(5):  # warmup
        model(dummy_data[0])

    max_iter = 400
    timer = Timer()
    with tqdm.tqdm(total=max_iter) as pbar:
        for idx, d in enumerate(f()):
            if idx == max_iter:
                break
            model(d)
            pbar.update()
    logger.info("{} iters in {} seconds.".format(max_iter, timer.seconds()))


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    args = parser.parse_args()
    assert not args.eval_only

    if args.task == "data":
        f = benchmark_data
    elif args.task == "train":
        """
        Note: training speed may not be representative.
        The training cost of a R-CNN model varies with the content of the data
        and the quality of the model.
        """
        f = benchmark_train
    elif args.task == "eval":
        f = benchmark_eval
        # only benchmark single-GPU inference.
        assert args.num_gpus == 1 and args.num_machines == 1
    launch(f, args.num_gpus, args.num_machines, args.machine_rank, args.dist_url, args=(args,))
