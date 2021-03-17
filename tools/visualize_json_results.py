#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
from collections import defaultdict
import tqdm

import cv2
import numpy as np

from cvpods.data import build_dataset
from cvpods.structures import Boxes, BoxMode, Instances
from cvpods.utils import PathManager, Visualizer, dynamic_import, setup_logger


def setup_cfg(path, logger):
    # load config from file and command-line arguments
    assert path.endswith(".py")
    path, module = os.path.split(path)
    module = module.rstrip(".py")
    cfg = dynamic_import(module, path).config
    if cfg.DATASETS.CUSTOM_TYPE != ["ConcatDataset", dict()]:
        logger.warning("Ignore cfg.DATASETS.CUSTOM_TYPE: {}. "
                       "Using (\"ConcatDataset\", dict())".format(cfg.DATASETS.CUSTOM_TYPE))
    cfg.DATASETS.CUSTOM_TYPE = ["ConcatDataset", dict()]

    return cfg


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])

    if score.shape[0] == 0:
        bbox = np.zeros((0, 4))
    else:
        bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--config", required=True,
                        help="path to a python file with a definition of `config`")
    parser.add_argument("--dataset",
                        help="name of the dataset. Use DATASETS.TEST[0] if not specified.",
                        default="")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()
    cfg = setup_cfg(args.config, logger)
    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    # TODO: add DatasetCatalog, MetadataCatalog
    dataset = build_dataset(
        cfg,
        [args.dataset] if args.dataset else [cfg.DATASETS.TEST[0]],
        transforms=[],
        is_train=False)
    dicts = dataset.datasets[0].dataset_dicts
    metadata = dataset.meta
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)

    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
