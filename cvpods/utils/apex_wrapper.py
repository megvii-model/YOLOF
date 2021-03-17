#!/usr/bin/python3
# -*- coding:utf-8 -*-

import functools

try:
    from apex import amp
except ImportError:
    pass


def is_amp_training():
    """
    check weather amp training is enabled.

    Returns:
        bool: True if amp training is enabled
    """
    try:
        return hasattr(amp._amp_state, "loss_scalers")
    except Exception:
        return False


def float_function(func):

    @functools.wraps(func)
    def float_wraps(*args, **kwargs):
        if is_amp_training():
            return amp.float_function(func)(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return float_wraps
