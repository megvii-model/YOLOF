# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by BaseDetection, Inc. and its affiliates.

import logging
import math
import os
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.data import build_test_loader, build_train_loader
from cvpods.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results
)
from cvpods.modeling.nn_utils.module_converter import maybe_convert_module
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import comm, setup_logger
from cvpods.utils.dump.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

from . import hooks
from .base_runner import RUNNERS, SimpleRunner

logger = logging.getLogger(__name__)


@RUNNERS.register()
class DefaultRunner(SimpleRunner):
    """
    A runner with default training logic. It does the following:

    1. Create a :class:`DefaultRunner` using model, optimizer, dataloader
       defined by the given config. Create a LR scheduler defined by the config.
    2. Load the last checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks defined by the config.

    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`DefaultRunner` are too much for research.

    The code of this class has been annotated about restrictive assumptions it makes.
    When they do not work for you, you're encouraged to:

    1. Overwrite methods of this class, OR:
    2. Use :class:`DefaultRunner`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.

    See the :doc:`/tutorials/training` tutorials for more details.

    Note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in cvpods.
    To obtain more stable behavior, write your own training logic with other public APIs.

    Examples:
    ::
        runner = DefaultRunner(cfg)
        runner.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        runner.train()

    Attributes:
        scheduler:
        checkpointer (DefaultCheckpointer):
        cfg (config dict):
    """

    def __init__(self, cfg, build_model):
        """
        Args:
            cfg (config dict):
        """
        logger = logging.getLogger("cvpods")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for cvpods
            setup_logger()
        self.logger = logger

        self.data_loader = self.build_train_loader(cfg)
        # Assume these objects must be constructed in this order.
        model = build_model(cfg)
        self.model = maybe_convert_module(model)
        self.logger.info(f"Model: \n{self.model}")

        # Assume these objects must be constructed in this order.
        self.optimizer = self.build_optimizer(cfg, self.model)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            if cfg.TRAINER.FP16.ENABLED:
                self.mixed_precision = True
                if cfg.TRAINER.FP16.TYPE == "APEX":
                    from apex import amp
                    self.model, self.optimizer = amp.initialize(
                        self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                    )
            else:
                self.mixed_precision = False
            torch.cuda.set_device(comm.get_local_rank())
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True)

        super().__init__(
            self.model,
            self.data_loader,
            self.optimizer,
        )

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            self.logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume = resume
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        if self.max_epoch is not None:
            self.start_epoch = self.start_iter // len(self.data_loader)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg
        # cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.OptimizationHook(
                accumulate_grad_steps=cfg.SOLVER.BATCH_SUBDIVISIONS,
                grad_clipper=None,
                mixed_precision=cfg.TRAINER.FP16.ENABLED
            ),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(),
                period=self.window_size))
        return ret

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`JSONWriter`.

        Args:
            output_dir: directory to store JSON metrics and tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(
                self.max_iter,
                window_size=self.window_size,
                epoch=self.max_epoch,
            ),
            JSONWriter(
                os.path.join(self.cfg.OUTPUT_DIR, "metrics.json"),
                window_size=self.window_size
            ),
            TensorboardXWriter(
                self.cfg.OUTPUT_DIR,
                window_size=self.window_size
            ),
        ]

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        if self.max_epoch is None:
            logger.info("Starting training from iteration {}".format(self.start_iter))
        else:
            logger.info("Starting training from epoch {}".format(self.start_epoch))

        super().train(self.start_iter, self.start_epoch, self.max_iter)

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`cvpods.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, **kwargs):
        """
        It now calls :func:`cvpods.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer, **kwargs)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`cvpods.data.build_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_test_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        raise NotImplementedError(
            """
If you want DefaultRunner to automatically run evaluation,
please implement `build_evaluator()` in subclasses (see train_net.py for example).
Alternatively, you can call evaluation functions yourself (see Colab balloon tutorial for example).
            """
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None, output_folder=None):
        """
        Args:
            cfg (config dict):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(
                        cfg, dataset_name, data_loader.dataset, output_folder=output_folder)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method.")
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def auto_scale_config(cfg, dataloader):

    logger = logging.getLogger(__name__)

    max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
    max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER

    subdivision = cfg.SOLVER.BATCH_SUBDIVISIONS
    # adjust lr by batch_subdivisions
    cfg.SOLVER.OPTIMIZER.BASE_LR *= subdivision

    """
    Here we use batch size * subdivision to simulator large batch training
    """
    if max_epoch:
        epoch_iter = math.ceil(
            len(dataloader.dataset) / (cfg.SOLVER.IMS_PER_BATCH * subdivision))

        if max_iter is not None:
            logger.warning(
                f"Training in EPOCH mode, automatically convert {max_epoch} epochs "
                f"into {max_epoch*epoch_iter} iters...")

        cfg.SOLVER.LR_SCHEDULER.MAX_ITER = max_epoch * epoch_iter
        cfg.SOLVER.LR_SCHEDULER.STEPS = [
            x * epoch_iter for x in cfg.SOLVER.LR_SCHEDULER.STEPS
        ]
        cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS = int(
            cfg.SOLVER.LR_SCHEDULER.WARMUP_ITERS * epoch_iter)
        cfg.SOLVER.CHECKPOINT_PERIOD = epoch_iter * cfg.SOLVER.CHECKPOINT_PERIOD
        cfg.TEST.EVAL_PERIOD = epoch_iter * cfg.TEST.EVAL_PERIOD
    else:
        epoch_iter = -1

    cfg.SOLVER.LR_SCHEDULER.EPOCH_ITERS = epoch_iter
