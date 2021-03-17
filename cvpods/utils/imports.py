#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import imp


def dynamic_import(config_name, config_path):
    """
    Dynamic import a project.

    Args:
        config_name (str): module name
        config_path (str): the dir that contains the .py with this module.

    Examples::
        >>> root = "/path/to/your/retinanet/"
        >>> project = root + "retinanet.res50.fpn.coco.800size.1x.mrcnn_sigmoid"
        >>> cfg = dynamic_import("config", project).config
        >>> net = dynamic_import("net", project)
    """
    fp, pth, desc = imp.find_module(config_name, [config_path])

    return imp.load_module(config_name, fp, pth, desc)
