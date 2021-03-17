#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

from .base_detection_config import BaseDetectionConfig

_config_dict = dict(
    MODEL=dict(
        PIXEL_MEAN=(0.485, 0.456, 0.406),
        PIXEL_STD=(0.229, 0.224, 0.225),
        DARKNET=dict(
            DEPTH=53,
            STEM_OUT_CHANNELS=32,
            WEIGHTS="cvpods/ImageNetPretrained/custom/darknet53.mix.pth",
            OUT_FEATURES=["dark3", "dark4", "dark5"]
        ),
        YOLO=dict(
            CLASSES=80,
            IN_FEATURES=["dark3", "dark4", "dark5"],
            ANCHORS=[
                [[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [42, 119]],
                [[10, 13], [16, 30], [33, 23]],
            ],
            CONF_THRESHOLD=0.01,  # TEST
            NMS_THRESHOLD=0.5,
            IGNORE_THRESHOLD=0.7,
        ),
    ),
)


class YOLO3Config(BaseDetectionConfig):
    def __init__(self, d=None, **kwargs):
        super().__init__(d, **kwargs)
        self._register_configuration(_config_dict)


config = YOLO3Config()
