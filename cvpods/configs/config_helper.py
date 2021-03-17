#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import logging
import re
import time
import warnings
from ast import literal_eval
from colorama import Back, Fore, Style
from easydict import EasyDict


def highlight(keyword, target, color=Fore.BLACK + Back.YELLOW):
    """
    use given color to highlight keyword in target string

    Args:
        keyword(str): highlight string
        target(str): target string
        color(str): string represent the color, use black foreground
        and yellow background as default

    Returns:
        (str) target string with keyword highlighted

    """
    return re.sub(keyword, color + r"\g<0>" + Style.RESET_ALL, target)


def find_key(param_dict: dict, key: str) -> dict:
    """
    find key in dict

    Args:
        param_dict(dict):
        key(str):

    Returns:
        (dict)

    Examples::
        >>> d = dict(abc=2, ab=4, c=4)
        >>> find_key(d, "ab")
        {'abc': 2, 'ab':4}

    """
    find_result = {}
    for k, v in param_dict.items():
        if re.search(key, k):
            find_result[k] = v
        if isinstance(v, dict):
            res = find_key(v, key)
            if res:
                find_result[k] = res
    return find_result


def diff_dict(src, dst):
    """
    find difference between src dict and dst dict

    Args:
        src(dict): src dict
        dst(dict): dst dict

    Returns:
        (dict) dict contains all the difference key

    """
    diff_result = {}
    for k, v in src.items():
        if k not in dst:
            diff_result[k] = v
        elif dst[k] != v:
            if isinstance(v, dict):
                diff_result[k] = diff_dict(v, dst[k])
            else:
                diff_result[k] = v
    return diff_result


def version_update(config):
    """
    Backward compatibility of old config's Augmentation pipelines and Optimizer configs;
    Convert old format into new ones.
    """
    # Old Augmentation config
    input_cfg = config.INPUT
    train_input_pop_list = [
        ("MIN_SIZE_TRAIN", "short_edge_length"),
        ("MIN_SIZE_TRAIN_SAMPLING", "sample_style"),
        ("MAX_SIZE_TRAIN", "max_size")
    ]
    test_input_pop_list = [
        ("MIN_SIZE_TEST", "short_edge_length"),
        ("MAX_SIZE_TEST", "max_size"),
    ]
    train_contains = [k for k in train_input_pop_list if k[0] in input_cfg]
    test_contains = [k for k in test_input_pop_list if k[0] in input_cfg]
    train_transforms = [t[0] for t in input_cfg["AUG"]["TRAIN_PIPELINES"]]
    test_transforms = [t[0] for t in input_cfg["AUG"]["TEST_PIPELINES"]]

    warnings.filterwarnings(
        "default",
        category=DeprecationWarning,
        module=__name__
    )  # only change local warning level
    if train_contains:
        warnings.warn("Old format training config will be deprecated ", DeprecationWarning)
        time.sleep(1)
    for k in train_contains:
        if "ResizeShortestEdge" in train_transforms:
            idx = idx = train_transforms.index("ResizeShortestEdge")
            input_cfg["AUG"]["TRAIN_PIPELINES"][idx][1][k[1]] = input_cfg[k[0]]

    if test_contains:
        warnings.warn("Old format testing config will be deprecated ", DeprecationWarning)
        time.sleep(1)
    for k in test_contains:
        if "ResizeShortestEdge" in test_transforms:
            idx = test_transforms.index("ResizeShortestEdge")
            input_cfg["AUG"]["TEST_PIPELINES"][idx][1][k[1]] = input_cfg[k[0]]

    for elem in train_contains + test_contains:
        config.INPUT.pop(elem[0])

    # Old SOLVER format
    solver_cfg = config.SOLVER
    candidates = list(solver_cfg.keys())
    lr_pop_list = [
        "LR_SCHEDULER_NAME", "MAX_ITER", "STEPS", "WARMUP_FACTOR", "WARMUP_ITERS",
        "WARMUP_METHOD", "GAMMA"
    ]
    optim_pop_list = [
        "BASE_LR", "BIAS_LR_FACTOR", "WEIGHT_DECAY", "WEIGHT_DECAY_NORM", "WEIGHT_DECAY_BIAS",
        "MOMENTUM"
    ]
    if any(item in candidates for item in lr_pop_list + optim_pop_list):
        warnings.warn("Old format solver config will be deprecated ", DeprecationWarning)
        time.sleep(1)
        if "LR_SCHEDULER" in solver_cfg and "OPTIMIZER" in solver_cfg:
            for k in candidates:
                if k in lr_pop_list:
                    solver_cfg["LR_SCHEDULER"][k] = solver_cfg[k]
                elif k in optim_pop_list:
                    solver_cfg["OPTIMIZER"][k] = solver_cfg[k]
        else:
            solver_cfg["LR_SCHEDULER"] = dict(
                NAME=solver_cfg.get("LR_SCHEDULER_NAME", "WarmupMultiStepLR"),
                MAX_ITER=solver_cfg.get("MAX_ITER", 30000),
                STEPS=solver_cfg.get("STEPS", (20000,)),
                WARMUP_FACTOR=solver_cfg.get("WARMUP_FACTOR", 1.0 / 1000),
                WARMUP_ITERS=solver_cfg.get("WARMUP_ITERS", 1000),
                WARMUP_METHOD=solver_cfg.get("WARMUP_METHOD", "linear"),
                GAMMA=solver_cfg.get("GAMMA", 0.1),
            )
            solver_cfg["OPTIMIZER"] = dict(
                NAME="SGD",
                BASE_LR=solver_cfg.get("BASE_LR", 0.001),
                BIAS_LR_FACTOR=solver_cfg.get("BIAS_LR_FACTOR", 1.0),
                WEIGHT_DECAY=solver_cfg.get("WEIGHT_DECAY", 0.0001),
                WEIGHT_DECAY_NORM=solver_cfg.get("WEIGHT_DECAY_NORM", 0.0),
                WEIGHT_DECAY_BIAS=solver_cfg.get("WEIGHT_DECAY_BIAS", 0.0001),
                MOMENTUM=solver_cfg.get("MOMENTUM", 0.9),
            )

        for k in candidates:
            if k in lr_pop_list + optim_pop_list:
                solver_cfg.pop(k)


def _assert_with_logging(cond, msg):
    logger = logging.getLogger(__name__)

    if not cond:
        logger.error(msg)
    assert cond, msg


def _decode_cfg_value(value):
    """
    Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    If the value is a dict, it will be interpreted as a new config dict.
    If the value is a str, it will be evaluated as literals.
    Otherwise it is returned as-is.

    Args:
        value (dict or str): value to be decoded
    """
    if isinstance(value, str):
        # Try to interpret `value` as a string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except (ValueError, SyntaxError):
            pass

    if isinstance(value, dict):
        return EasyDict(value)
    else:
        return value


def _cast_cfg_value_type(replacement, original, full_key):
    """
    Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    logger = logging.getLogger(__name__)
    ori_type = type(original)
    new_type = type(replacement)

    if original is None or replacement is None:
        logger.info("None type, {} to {}".format(ori_type, new_type))
        return replacement

    # The types must match (with some exceptions)
    if new_type == ori_type:
        logger.info(
            "change value of '{}' from {} to {}".format(full_key, original, replacement)
        )
        return replacement

    # try to casts replacement to original type
    try:
        replacement = ori_type(replacement)
        return replacement
    except Exception:
        logger.error(
            "Could not cast '{}' from {} to {} with values ({} vs. {})".format(
                full_key, new_type, ori_type, replacement, original)
        )
        raise ValueError
