#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

import torch

from cvpods.configs.base_config import config as cfg
from cvpods.utils import Visualizer
from cvpods.utils.visualizer.show import visualize_feature_maps


def basenet(cls):

    def data_analyze_on(self):
        if not hasattr(cls, 'analyze_buffer'):
            cls.analyze_buffer = defaultdict(list)

    cls.data_analyze_on = data_analyze_on

    def visualize_data(self, per_image, save_to_file=False):
        """
        Visualize data from batch_inputs of dataloader.

        Args:
            per_image (dict): a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
            save_to_file: whether save img to disk.

        Example:
            >>> self.visualize_data(batch_inputs[0])
        """
        metadata = self.data_meta

        def output(vis, fname):
            if not save_to_file:
                print(fname)
                cv2.imshow("window", vis.get_image()[:, :, ::-1])
                cv2.waitKey()
            else:
                filepath = os.path.join("./", fname)
                print("Saving to {} ...".format(filepath))
                vis.save(filepath)

        scale = 1.0
        # Pytorch tensor is in (C, H, W) format
        img = per_image["image"].permute(1, 2, 0)
        if cfg.INPUT.FORMAT == "BGR":
            img = img[:, :, [2, 1, 0]]
        else:
            img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        target_fields = per_image["instances"].get_fields()
        labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=target_fields.get("gt_masks", None),
            keypoints=target_fields.get("gt_keypoints", None),
        )
        output(vis, str(per_image["image_id"]) + ".jpg")

    cls.visualize_data = visualize_data

    def visualize_predict_data(self, per_image, per_instalce, save_to_file=False):
        metadata = self.data_meta

        def output(vis, fname):
            if not save_to_file:
                print(fname)
                cv2.imshow("window", vis.get_image()[:, :, ::-1])
                cv2.waitKey()
            else:
                filepath = os.path.join("./", fname)
                print("Saving to {} ...".format(filepath))
                vis.save(filepath)

        scale = 1.0
        # Pytorch tensor is in (C, H, W) format
        img = per_image["image"].permute(1, 2, 0)
        if cfg.INPUT.FORMAT == "BGR":
            img = img[:, :, [2, 1, 0]]
        else:
            img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

        visualizer = Visualizer(img, metadata=metadata, scale=scale)
        vis = visualizer.draw_instance_predictions(per_instalce)
        output(vis, str(per_image["image_id"]) + ".jpg")

    cls.visualize_predict_data = visualize_predict_data

    def visualize_feature_map(self, feature_map, per_image=None, stride=8,
                              save_name=0, with_img=True, channelwise=False):
        """
        Visualize feature map with (optional) gt boxes

        Args:
            feature_map (torch.Tensor): C x H x W
            per_image (dict): batch_inputs[i]
            stride (int): down sample ratio of current feature_map
            save_name (int or str): feature map figure name
            with_img (bool): weather visualize corresponding image data
            channelwise (bool): visualize feature map mean or all channels

        Examples::
            >>> level = 1
            >>> self.visualize_feature_map(features[level][0],
            >>>                        per_image=batched_inputs[level],
            >>>                        stride=self.fpn_strides[level],
            >>>                        save_name=1,
            >>>                        with_img=False,
            >>>                        channelwise=False)
        """
        if with_img and save_name == 0:
            self.visualize_data(per_image)

        with torch.no_grad():
            if "instances" in per_image:
                instance = per_image["instances"]
                gts = instance.gt_boxes.tensor.cpu().numpy()
                l = gts[:, 0:1]  # noqa:E741
                t = gts[:, 1:2]
                r = gts[:, 2:3]
                b = gts[:, 3:4]
                boxes = (
                    np.concatenate([l, t, l, b, r, b, r, t], axis=1)
                    .reshape(-1, 4, 2)
                    .transpose(0, 2, 1)
                )
            else:
                boxes = []
            if not channelwise:
                fm = feature_map.permute(1, 2, 0).mean(dim=-1, keepdim=True)
            else:
                fm = feature_map.permute(1, 2, 0)
            # visualize_feature_maps(fm.sigmoid().cpu().numpy(),
            visualize_feature_maps(
                fm.cpu().numpy(),
                boxes=boxes,
                stride=stride,
                save_filename=f"feature_map_{save_name}.png",
            )

    cls.visualize_feature_map = visualize_feature_map

    return cls
