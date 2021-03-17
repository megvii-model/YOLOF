#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import collections
import os
import pprint
import re
import six
from colorama import Back, Fore
from easydict import EasyDict
from tabulate import tabulate

from .config_helper import (
    _assert_with_logging,
    _cast_cfg_value_type,
    _decode_cfg_value,
    diff_dict,
    find_key,
    highlight,
    version_update
)

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except ImportError:
    collectionsAbc = collections


_config_dict = dict(
    MODEL=dict(
        DEVICE="cuda",
        # Path (possibly with schema like catalog://, detectron2://, s3://) to a checkpoint file
        # to be loaded to the model. You can find available models in the model zoo.
        WEIGHTS="",
        # Indicate whether convert final checkpoint to use as pretrain weights of preceeding model
        AS_PRETRAIN=False,
        # Values to be used for image normalization (BGR order, since INPUT.FORMAT defaults to BGR)
        # To train on images of different number of channels, just set different mean & std.
        PIXEL_MEAN=[103.530, 116.280, 123.675],
        # When using pre-trained models in Detectron1 or any MSRA models,
        # std has been absorbed into its conv1 weights, so the std needs to be
        # set 1. Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
        PIXEL_STD=[1.0, 1.0, 1.0],
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=(800,), max_size=1333, sample_style="choice")),
                ("RandomFlip", dict()),
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge", dict(
                    short_edge_length=800, max_size=1333, sample_style="choice")),
            ],
        ),
        # Whether the model needs RGB, YUV, HSV etc.
        FORMAT="BGR",
        # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
        MASK_FORMAT="polygon",
    ),
    DATASETS=dict(
        CUSTOM_TYPE=("ConcatDataset", dict()),
        # List of the dataset names for training.
        # Must be registered in cvpods/data/datasets/paths_route
        TRAIN=(),
        # List of the pre-computed proposal files for training, which must be consistent
        # with datasets listed in DATASETS.TRAIN.
        PROPOSAL_FILES_TRAIN=(),
        # Number of top scoring precomputed proposals to keep for training
        PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
        # List of the dataset names for testing.
        # Must be registered in cvpods/data/datasets/paths_route
        TEST=(),
        # List of the pre-computed proposal files for test, which must be consistent
        # with datasets listed in DATASETS.TEST.
        PROPOSAL_FILES_TEST=(),
        # Number of top scoring precomputed proposals to keep for test
        PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    ),
    DATALOADER=dict(
        # Number of data loading threads
        NUM_WORKERS=2,
        # If True, each batch should contain only images for which the aspect ratio
        # is compatible. This groups portrait images together, and landscape images
        # are not batched with portrait images.
        ASPECT_RATIO_GROUPING=True,
        # Default sampler for dataloader
        SAMPLER_TRAIN="DistributedGroupSampler",
        # Repeat threshold for RepeatFactorTrainingSampler
        REPEAT_THRESHOLD=0.0,
        # If True, the dataloader will filter out images that have no associated
        # annotations at train time.
        FILTER_EMPTY_ANNOTATIONS=True,
    ),
    SOLVER=dict(
        # Configs of lr scheduler
        LR_SCHEDULER=dict(
            NAME="WarmupMultiStepLR",
            MAX_ITER=40000,
            MAX_EPOCH=None,
            # STEPS supports both iterations and epochs.
            # If MAX_EPOCH are specified, STEPS will be calculated automatically
            STEPS=(30000,),
            WARMUP_FACTOR=1.0 / 1000,
            WARMUP_ITERS=1000,
            # WARMUP_METHOD in "linear", "constant", "brunin"
            WARMUP_METHOD="linear",
            # Decrease learning rate by GAMMA.
            GAMMA=0.1,
        ),
        OPTIMIZER=dict(
            NAME="D2SGD",
            BASE_LR=0.001,
            # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for biases.
            # This is not useful (at least for recent models). You should avoid
            # changing these and they exist only to reproduce Detectron v1 training if desired.
            BIAS_LR_FACTOR=1.0,
            WEIGHT_DECAY=0.0001,
            # The weight decay that's applied to parameters of normalization layers
            WEIGHT_DECAY_NORM=0.0,
            MOMENTUM=0.9,
        ),
        # Gradient clipping
        CLIP_GRADIENTS=dict(
            ENABLED=False,
            # - "value": the absolute values of elements of each gradients are clipped
            # - "norm": the norm of the gradient for each parameter is clipped thus
            #   affecting all elements in the parameter
            CLIP_TYPE="value",
            # Maximum absolute value used for clipping gradients
            CLIP_VALUE=1.0,
            # Floating point number p for L-p norm to be used with the "norm"
            # gradient clipping type; for L-inf, please specify .inf
            NORM_TYPE=2.0,
        ),
        # Save a checkpoint after every this number of iterations
        CHECKPOINT_PERIOD=5000,
        # Number of images per batch across all machines.
        # If we have 16 GPUs and IMS_PER_BATCH = 32,
        # each GPU will see 2 images per batch.
        IMS_PER_BATCH=16,
        IMS_PER_DEVICE=2,
        BATCH_SUBDIVISIONS=1,
    ),
    TEST=dict(
        # For end-to-end tests to verify the expected accuracy.
        # Each item is [task, metric, value, tolerance]
        # e.g.: [['bbox', 'AP', 38.5, 0.2]]
        EXPECTED_RESULTS=[],
        # The period (in terms of steps) to evaluate the model during training.
        # If using positive EVAL_PERIOD, every #EVAL_PERIOD iter will evaluate automaticly.
        # If EVAL_PERIOD = 0, model will be evaluated after training.
        # If using negative EVAL_PERIOD, no evaluation will be applied.
        EVAL_PERIOD=0,
        # The sigmas used to calculate keypoint OKS. See http://cocodataset.org/#keypoints-eval
        # When empty it will use the defaults in COCO.
        # Otherwise it should have the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
        KEYPOINT_OKS_SIGMAS=[],
        # Maximum number of detections to return per image during inference (100 is
        # based on the limit established for the COCO dataset).
        DETECTIONS_PER_IMAGE=100,
        AUG=dict(
            ENABLED=False,
            MIN_SIZES=(400, 500, 600, 700, 800, 900, 1000, 1100, 1200),
            MAX_SIZE=4000,
            FLIP=True,
            EXTRA_SIZES=(),
            SCALE_FILTER=False,
            SCALE_RANGES=(),
        ),
        PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    ),
    # Trainer is used to specify options related to control the training process
    TRAINER=dict(
        NAME="DefaultRunner",
        WINDOW_SIZE=20,
        FP16=dict(
            ENABLED=False,
            # options: [APEX, PyTorch]
            TYPE="APEX",
            # OPTS: kwargs for each option
            OPTS=dict(
                OPT_LEVEL="O1",
            ),
        ),
    ),
    # Directory where output files are written
    OUTPUT_DIR="./output",
    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed does not
    # guarantee fully deterministic behavior.
    SEED=-1,
    # Benchmark different cudnn algorithms.
    # If input images have very different sizes, this option will have large overhead
    # for about 10k iterations. It usually hurts total time, but can benefit for certain models.
    # If input images have the same or similar sizes, benchmark is often helpful.
    CUDNN_BENCHMARK=False,
    # The period (in terms of steps) for minibatch visualization at train time.
    # Set to 0 to disable.
    VIS_PERIOD=0,
    # global config is for quick hack purposes.
    # You can set them in command line or config files,
    # Do not commit any configs into it.
    GLOBAL=dict(
        HACK=1.0,
        DUMP_TRAIN=True,
        DUMP_TEST=False,
    ),
)


class ConfigDict(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if (
                not (k.startswith("__") and k.endswith("__"))
                and k not in self.funcname_not_in_attr()
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [EasyDict(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict):
            value = EasyDict(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def funcname_not_in_attr(self):
        return [
            "update", "pop", "merge",
            "merge_from_list", "find", "diff",
            "inner_dict", "funcname_not_in_attr"
        ]

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)

    def merge(self, config=None, **kwargs):
        """
        merge all key and values of config as BaseConfig's attributes.
        Note that kwargs will override values in config if they have the same keys

        Args:
            config (dict): custom config dict
        """
        def update_helper(d, u):
            for k, v in six.iteritems(u):
                dv = d.get(k, EasyDict())
                if not isinstance(dv, collectionsAbc.Mapping):
                    d[k] = v
                elif isinstance(v, collectionsAbc.Mapping):
                    d[k] = update_helper(dv, v)
                else:
                    d[k] = v
            return d

        if config is not None:
            update_helper(self, config)
        if kwargs:
            update_helper(self, kwargs)

    def merge_from_list(self, cfg_list):
        """
        Merge config (keys, values) in a list (e.g., from command line) into
        this config dict.

        Args:
            cfg_list (list): cfg_list must be divided exactly.
            For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
                d = d[subkey]
            subkey = key_list[-1]
            _assert_with_logging(subkey in d, "Non-existent key: {}".format(full_key))
            value = _decode_cfg_value(v)
            value = _cast_cfg_value_type(value, d[subkey], full_key)
            d[subkey] = value

    def diff(self, cfg) -> dict:
        """
        diff given config with current config object

        Args:
            cfg(ConfigDict): given config, could be any subclass of ConfigDict

        Returns:
            ConfigDict: contains all diff pair
        """
        assert isinstance(cfg, ConfigDict), "config is not a subclass of ConfigDict"
        diff_result = {}
        conf_keys = cfg.keys()
        for param in self.keys():
            if param not in conf_keys:
                diff_result[param] = getattr(self, param)
            else:
                self_val, conf_val = getattr(self, param), getattr(cfg, param)
                if self_val != conf_val:
                    if isinstance(self_val, dict) and isinstance(conf_val, dict):
                        diff_result[param] = diff_dict(self_val, conf_val)
                    else:
                        diff_result[param] = self_val
        return ConfigDict(diff_result)

    def find(self, key: str, show=True, color=Fore.BLACK + Back.YELLOW) -> dict:
        """
        find a given key and its value in config

        Args:
            key (str): the string you want to find
            show (bool): if show is True, print find result; or return the find result
            color (str): color of `key`, default color is black(foreground) yellow(background)

        Returns:
            dict: if  show is False, return dict that contains all find result

        Example::

            >>> from config import config        # suppose you are in your training dir
            >>> config.find("weights")
        """
        key = key.upper()
        find_result = {}
        for param, param_value in self.items():
            param_value = getattr(self, param)
            if re.search(key, param):
                find_result[param] = param_value
            elif isinstance(param_value, dict):
                find_res = find_key(param_value, key)
                if find_res:
                    find_result[param] = find_res
        if not show:
            return find_result
        else:
            pformat_str = repr(ConfigDict(find_result))
            print(highlight(key, pformat_str, color))

    def __repr__(self):
        param_list = [(k, pprint.pformat(v)) for k, v in self.items()]
        table_header = ["config params", "values"]
        return tabulate(param_list, headers=table_header, tablefmt="fancy_grid")


class BaseConfig(ConfigDict):

    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)

    def _register_configuration(self, config):
        self.merge(config)
        version_update(self)

    def link_log(self, link_name="log"):
        """
        create a softlink to output dir.

        Args:
            link_name(str): name of softlink
        """
        if os.path.islink(link_name) and os.readlink(link_name) != self.OUTPUT_DIR:
            os.system("rm " + link_name)
        if not os.path.exists(link_name):
            cmd = "ln -s {} {}".format(self.OUTPUT_DIR, link_name)
            os.system(cmd)

    def funcname_not_in_attr(self):
        namelist = super().funcname_not_in_attr()
        namelist.extend(["link_log", "_register_configuration"])
        return namelist


config = BaseConfig()
