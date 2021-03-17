#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) BaseDetection, Inc. and its affiliates.

import math

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvpods.data.transforms.transform_gen import CenterAffine
from cvpods.layers import DeformConvWithOff, ModulatedDeformConvWithOff, ShapeSpec
from cvpods.modeling.losses import reg_l1_loss
from cvpods.modeling.nn_utils.feature_utils import gather_feature
from cvpods.structures import Boxes, ImageList, Instances


class DeconvLayer(nn.Module):

    def __init__(
        self, in_planes,
        out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
    ):
        super(DeconvLayer, self).__init__()
        if modulate_deform:
            self.dcn = ModulatedDeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        else:
            self.dcn = DeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )

        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class CenternetDeconv(nn.Module):

    def __init__(self, cfg):
        super(CenternetDeconv, self).__init__()
        # modify into config
        channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL
        deconv_kernel = cfg.MODEL.CENTERNET.DECONV_KERNEL
        modulate_deform = cfg.MODEL.CENTERNET.MODULATE_DEFORM
        self.deconv1 = DeconvLayer(
            channels[0], channels[1],
            deconv_kernel=deconv_kernel[0],
            modulate_deform=modulate_deform,
        )
        self.deconv2 = DeconvLayer(
            channels[1], channels[2],
            deconv_kernel=deconv_kernel[1],
            modulate_deform=modulate_deform,
        )
        self.deconv3 = DeconvLayer(
            channels[2], channels[3],
            deconv_kernel=deconv_kernel[2],
            modulate_deform=modulate_deform,
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class CenterNet(nn.Module):
    r"""
    Implement CenterNet (https://arxiv.org/abs/1904.07850).
    """
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cfg = cfg

        # fmt: off
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        # Loss parameters:
        # Inference parameters:
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on
        self.backbone = cfg.build_backbone(
            cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))
        )
        self.upsample = CenternetDeconv(cfg)
        self.head = CenternetHead(cfg)
        self.reg_loss = reg_l1_loss()

        self.mean, self.std = cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD
        pixel_mean = torch.Tensor(self.mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(self.std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        r"""
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            dict[str: Tensor]:
        """
        images = self.preprocess_image(batched_inputs)

        if not self.training:
            return self.inference(images)

        features = self.backbone(images.tensor)
        up_fmap = self.upsample(features["res5"])
        pred_dict = self.head(up_fmap)

        gt_dict = self.get_ground_truth(batched_inputs)

        return self.losses(pred_dict, gt_dict)

    def losses(self, pred_dict, gt_dict):
        r"""
        calculate losses of pred and gt

        Args:
            gt_dict (dict): a dict contains all information of gt
                gt_dict = {
                    "score_map": gt scoremap,
                    "wh": gt width and height of boxes,
                    "reg": gt regression of box center point,
                    "reg_mask": mask of regression,
                    "index": gt index,
                }
            pred (dict): a dict contains all information of prediction
                pred = {
                    "cls": predicted score map,
                    "reg": predcited regression,
                    "wh": predicted width and height of box,
                }
        """
        # scoremap loss
        pred_score = pred_dict['cls']
        cur_device = pred_score.device
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(cur_device)

        loss_cls = _modified_focal_loss(pred_score, gt_dict['score_map'])

        mask = gt_dict['reg_mask']
        index = gt_dict['index']
        index = index.to(torch.long)
        # width and height loss, better version
        loss_wh = self.reg_loss(pred_dict['wh'], mask, index, gt_dict['wh'])

        # regression loss
        loss_reg = self.reg_loss(pred_dict['reg'], mask, index, gt_dict['reg'])

        loss_cls *= self.cfg.MODEL.LOSS.CLS_WEIGHT
        loss_wh *= self.cfg.MODEL.LOSS.WH_WEIGHT
        loss_reg *= self.cfg.MODEL.LOSS.REG_WEIGHT

        loss = {
            "loss_cls": loss_cls,
            "loss_box_wh": loss_wh,
            "loss_center_reg": loss_reg,
        }
        return loss

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        return CenterNetGT.generate(self.cfg, batched_inputs)

    @torch.no_grad()
    def inference(self, images):
        r"""
        image(tensor): ImageList in cvpods.structures
        """
        n, c, h, w = images.tensor.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.cfg.MODEL.CENTERNET.DOWN_SCALE
        img_info = dict(center=center_wh, size=size_wh,
                        height=new_h // down_scale,
                        width=new_w // down_scale)

        pad_value = [-x / y for x, y in zip(self.mean, self.std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(images.tensor.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h:h + pad_h, pad_w:w + pad_w] = images.tensor

        features = self.backbone(aligned_img)
        up_fmap = self.upsample(features["res5"])
        pred_dict = self.head(up_fmap)
        results = self.decode_prediction(pred_dict, img_info)

        ori_w, ori_h = img_info['center'] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **results)

        return [{"instances": det_instance}]

    def decode_prediction(self, pred_dict, img_info):
        r"""
        Args:
            pred_dict (dict): a dict contains all information of prediction
            img_info (dict): a dict contains needed information of origin image
        """
        fmap = pred_dict["cls"]
        reg = pred_dict["reg"]
        wh = pred_dict["wh"]

        boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)
        # boxes = Boxes(boxes.reshape(boxes.shape[-2:]))
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)

        # dets = CenterNetDecoder.decode(fmap, wh, reg)
        boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
        boxes = Boxes(boxes)
        return dict(pred_boxes=boxes, scores=scores, pred_classes=classes)

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # images = [img / 255 for img in images]
        images = [self.normalizer(img / 255.0) for img in images]
        images = ImageList.from_tensors(images,
                                        self.backbone.size_divisibility)
        return images


class CenterNetDecoder(object):

    @staticmethod
    def decode(fmap, wh, reg=None, cat_spec_wh=False, K=100):
        r"""
        decode feature maps, width height, regression to detections results.

        Args:
            fmap (Tensor): input feature map.
            wh (Tensor): tensor represents (width, height).
            reg (Tensor): tensor represents regression.
            cat_spec_wh (bool): whether reshape wh tensor.
            K (int): top k value in score map.
        """
        batch, channel, height, width = fmap.shape

        fmap = CenterNetDecoder.pseudo_nms(fmap)

        scores, index, clses, ys, xs = CenterNetDecoder.topk_score(fmap, K=K)
        if reg is not None:
            reg = gather_feature(reg, index, use_transform=True)
            reg = reg.reshape(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = gather_feature(wh, index, use_transform=True)

        if cat_spec_wh:
            wh = wh.view(batch, K, channel, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            wh = wh.gather(2, clses_ind).reshape(batch, K, 2)
        else:
            wh = wh.reshape(batch, K, 2)

        clses  = clses.reshape(batch, K, 1).float()
        scores = scores.reshape(batch, K, 1)

        half_w = wh[..., 0:1] / 2
        half_h = wh[..., 1:2] / 2
        bboxes = torch.cat([xs - half_w, ys - half_h,
                            xs + half_w, ys + half_h], dim=2)

        detections = (bboxes, scores, clses)

        return detections

    @staticmethod
    def transform_boxes(boxes, img_info):
        r"""
        transform predicted boxes to target boxes

        Args:
            boxes (Tensor): torch Tensor with (Batch, N, 4) shape
            img_info (dict): dict contains all information of original image
        """
        boxes = boxes.cpu().numpy().reshape(-1, 4)

        center = img_info['center']
        size = img_info['size']
        output_size = (img_info['width'], img_info['height'])
        src, dst = CenterAffine.generate_src_and_dst(center, size, output_size)
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))

        coords = boxes.reshape(-1, 2)
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        target_boxes = np.dot(aug_coords, trans.T).reshape(-1, 4)
        return target_boxes

    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply maxpooling instead of NMS.

        Args:
            fmap (Tensor): output feature maps.
            pool_size (int): max pooling window size.
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    @staticmethod
    def topk_score(scores, K=40):
        r"""
        get top point in score map.

        Args:
            scores (Tensor): scores map.
            K (int): top K in scores map.
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class CenterNetGT(object):

    @staticmethod
    def generate(config, batched_input):
        r"""
        genterate ground truth for CenterNet
        """
        box_scale = 1 / config.MODEL.CENTERNET.DOWN_SCALE
        num_classes = config.MODEL.CENTERNET.NUM_CLASSES
        output_size = config.INPUT.OUTPUT_SIZE
        min_overlap = config.MODEL.CENTERNET.MIN_OVERLAP
        tensor_dim = config.MODEL.CENTERNET.TENSOR_DIM

        scoremap_list, wh_list, reg_list, reg_mask_list, index_list = [[] for i in range(5)]
        for data in batched_input:
            # img_size = (data['height'], data['width'])

            bbox_dict = data['instances'].get_fields()

            # init gt tensors
            gt_scoremap = torch.zeros(num_classes, *output_size)
            gt_wh = torch.zeros(tensor_dim, 2)
            gt_reg = torch.zeros_like(gt_wh)
            reg_mask = torch.zeros(tensor_dim)
            gt_index = torch.zeros(tensor_dim)
            # pass

            boxes, classes = bbox_dict['gt_boxes'], bbox_dict['gt_classes']
            num_boxes = boxes.tensor.shape[0]
            boxes.scale(box_scale, box_scale)

            centers = boxes.get_centers()
            centers_int = centers.to(torch.int32)
            gt_index[:num_boxes] = centers_int[..., 1] * output_size[1] + centers_int[..., 0]
            gt_reg[:num_boxes] = centers - centers_int
            reg_mask[:num_boxes] = 1

            wh = torch.zeros_like(centers)
            box_tensor = boxes.tensor
            wh[..., 0] = box_tensor[..., 2] - box_tensor[..., 0]
            wh[..., 1] = box_tensor[..., 3] - box_tensor[..., 1]
            CenterNetGT.generate_score_map(
                gt_scoremap, classes, wh,
                centers_int, min_overlap,
            )
            gt_wh[:num_boxes] = wh

            scoremap_list.append(gt_scoremap)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(reg_mask)
            index_list.append(gt_index)

        gt_dict = {
            "score_map": torch.stack(scoremap_list, dim=0),
            "wh": torch.stack(wh_list, dim=0),
            "reg": torch.stack(reg_list, dim=0),
            "reg_mask": torch.stack(reg_mask_list, dim=0),
            "index": torch.stack(index_list, dim=0),
        }
        return gt_dict

    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        r"""
        generate score map

        Args:
            fmap (Tensor): input feature map.
            gt_class (Tensor): tensor represents ground truth classes.
            gt_wh (Tensor): ground truth width and height value.
            centers_int (Tensor): ground truth int value of centers.
            min_overlap (float): IoU threshold.
        """
        radius = CenterNetGT.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetGT.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        r"""
        get gaussian radius according to box size and IoU threshold, copyed from CornerNet.

        box_size: (w, h) information. Could be a torch.Tensor, numpy.ndarray, list or tuple.
        NOTE: we are using a bug-version, please refer to fix bug version in CornerNet.
        """
        box_tensor = torch.Tensor(box_size)
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        r"""
        generate guassian distribution according to gaussian radius and sigma.

        Args:
            radius (Tensor): radius of gaussian radius.
            sigma (int): sigma in gaussian.
        """
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        r"""
        generate ground truth for CenterNet

        Args:
            fmap (Tensor): output feature map
            center (Tensor): gaussian center
            radius (Tensor): gaussian radius
            k (int): topk
        """
        diameter = 2 * radius + 1
        gaussian = CenterNetGT.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        # return fmap


def _modified_focal_loss(pred, gt):
    r"""
    focal loss used for CenterNet, modified from focal loss.
    but this function is a numeric stable version implementation.
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    pred = torch.max(pred, torch.ones_like(pred) * 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


class SingleHead(nn.Module):
    r"""
    Single head used in CenterNet Head.
    """

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenternetHead(nn.Module):
    r"""
    The head used in CenterNet for object classification and box regression.
    It has three single heads, with a common structure but separate parameters.
    """

    def __init__(self, cfg):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.wh_head = SingleHead(64, 2)
        self.reg_head = SingleHead(64, 2)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        return pred
