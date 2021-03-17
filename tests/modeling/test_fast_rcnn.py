# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import unittest
import torch

from cvpods.configs import RCNNConfig
from cvpods.modeling.box_regression import Box2BoxTransform, Box2BoxTransformRotated
from cvpods.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from cvpods.modeling.roi_heads.rotated_fast_rcnn import RotatedFastRCNNOutputs
from cvpods.structures import Boxes, Instances, RotatedBoxes
from cvpods.utils import EventStorage

logger = logging.getLogger(__name__)


class FastRCNNTest(unittest.TestCase):
    def test_fast_rcnn(self):
        torch.manual_seed(132)
        cfg = RCNNConfig()
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5)
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        box_head_output_size = 8
        num_classes = 5
        cls_agnostic_bbox_reg = False

        box_predictor = FastRCNNOutputLayers(
            box_head_output_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
        )
        feature_pooled = torch.rand(2, box_head_output_size)
        pred_class_logits, pred_proposal_deltas = box_predictor(feature_pooled)
        image_shape = (10, 10)
        proposal_boxes = torch.tensor([[0.8, 1.1, 3.2, 2.8], [2.3, 2.5, 7, 8]], dtype=torch.float32)
        gt_boxes = torch.tensor([[1, 1, 3, 3], [2, 2, 6, 6]], dtype=torch.float32)
        result = Instances(image_shape)
        result.proposal_boxes = Boxes(proposal_boxes)
        result.gt_boxes = Boxes(gt_boxes)
        result.gt_classes = torch.tensor([1, 2])
        proposals = []
        proposals.append(result)
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
        )
        with EventStorage():  # capture events in a new storage to discard them
            losses = outputs.losses()

        expected_losses = {
            "loss_cls": torch.tensor(1.7951188087),
            "loss_box_reg": torch.tensor(4.0357131958),
        }
        for name in expected_losses.keys():
            assert torch.allclose(losses[name], expected_losses[name])

    def test_fast_rcnn_empty_batch(self, device="cpu"):
        cfg = RCNNConfig()
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5)
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        logits = torch.randn(0, 100, requires_grad=True, device=device)
        deltas = torch.randn(0, 4, requires_grad=True, device=device)

        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        outputs = FastRCNNOutputs(
            box2box_transform, logits, deltas, [], smooth_l1_beta
        )
        with EventStorage():  # capture events in a new storage to discard them
            losses = outputs.losses()

        for value in losses.values():
            self.assertTrue(torch.allclose(value, torch.zeros_like(value)))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_fast_rcnn_empty_batch_cuda(self):
        self.test_fast_rcnn_empty_batch(device=torch.device("cuda"))

    def test_fast_rcnn_rotated(self):
        torch.manual_seed(132)
        cfg = RCNNConfig()
        cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10, 10, 5, 5, 1)
        box2box_transform = Box2BoxTransformRotated(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        box_head_output_size = 8
        num_classes = 5
        cls_agnostic_bbox_reg = False

        box_predictor = FastRCNNOutputLayers(
            box_head_output_size, num_classes, cls_agnostic_bbox_reg, box_dim=5
        )
        feature_pooled = torch.rand(2, box_head_output_size)
        pred_class_logits, pred_proposal_deltas = box_predictor(feature_pooled)
        image_shape = (10, 10)
        proposal_boxes = torch.tensor(
            [[2, 1.95, 2.4, 1.7, 0], [4.65, 5.25, 4.7, 5.5, 0]], dtype=torch.float32
        )
        gt_boxes = torch.tensor([[2, 2, 2, 2, 0], [4, 4, 4, 4, 0]], dtype=torch.float32)
        result = Instances(image_shape)
        result.proposal_boxes = RotatedBoxes(proposal_boxes)
        result.gt_boxes = RotatedBoxes(gt_boxes)
        result.gt_classes = torch.tensor([1, 2])
        proposals = []
        proposals.append(result)
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = RotatedFastRCNNOutputs(
            box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta
        )
        with EventStorage():  # capture events in a new storage to discard them
            losses = outputs.losses()

        # Note: the expected losses are slightly different even if
        # the boxes are essentially the same as in the FastRCNNOutput test, because
        # bbox_pred in FastRCNNOutputLayers have different Linear layers/initialization
        # between the two cases.
        expected_losses = {
            "loss_cls": torch.tensor(1.7920907736),
            "loss_box_reg": torch.tensor(4.0410838127),
        }
        for name in expected_losses.keys():
            assert torch.allclose(losses[name], expected_losses[name])


if __name__ == "__main__":
    unittest.main()
