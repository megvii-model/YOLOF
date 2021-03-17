#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy

import torch

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.data import build_transform_gens

__all__ = ["DefaultPredictor"]


class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """
    def __init__(self, cfg, meta):
        self.cfg = deepcopy(cfg)
        if self.cfg.MODEL.DEVICE.startswith("cuda:"):
            torch.cuda.set_device(self.cfg.MODEL.DEVICE)
            self.cfg.MODEL.DEVICE = "cuda"
        self.model = cfg.build_model(self.cfg)
        self.model.eval()
        self.metadata = meta

        checkpointer = DefaultCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = build_transform_gens(cfg.INPUT.AUG.TEST_PIPELINES)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad(
        ):  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]

            image = original_image
            for tfm_gen in self.transform_gen:
                image = tfm_gen.get_transform(image).apply_image(image)

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
