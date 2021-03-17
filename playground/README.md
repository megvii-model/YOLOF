- [cvpods_playground](#cvpods_playground)
  * [ImageNet Classification](#imagenet-classification)
  * [Self\-supervised Learning](#self--supervised-learning)
  * [Object Detection](#object-detection)
    + [COCO](#coco)
      - [Faster R-CNN](#faster-r-cnn)
      - [RetinaNet](#retinanet)
      - [FCOS](#fcos)
      - [ATSS](#atss)
      - [FreeAnchor](#freeanchor)
      - [TridentNet](#tridentnet)
      - [RepPoints](#reppoints)
      - [CenterNet](#centernet)
      - [EfficientDet](#efficientdet)
      - [YOLO](#yolo)
      - [SSD](#ssd)
      - [DETR](#detr)
      - [Sparse R-CNN](#sparse-r-cnn)
    + [PASCAL VOC](#pascal-voc)
    + [WIDER FACE](#wider-face)
    + [CityPersons](#citypersons)
    + [CrowdHuman](#crowdhuman)
  * [Instance Segmentation](#instance-segmentation)
    + [COCO](#coco-1)
      - [Mask R-CNN](#mask-r-cnn)
      - [TensorMask](#tensormask)
      - [CascadeRCNN](#cascadercnn)
      - [PointRend](#pointrend)
      - [SOLO](#solo)
    + [LVIS](#lvis)
    + [CITYSCAPES](#cityscapes)
  * [Semantic Segmentation](#semantic-segmentation)
    + [COCO](#coco-2)
      - [SemanticFPN](#semanticfpn)
    + [CITYSCAPES](#cityscapes-1)
      - [PointRend](#pointrend-1)
      - [DynamicRouting](#dynamicrouting)
      - [FCN](#fcn)
  * [Panoptic Segmentation](#panoptic-segmentation)
    + [COCO](#coco-3)
      - [PanopticFPN](#Panopticfpn)
  * [Key\-Points](#key-points)
    + [COCO_PERSON](#coco_person)
      - [Keypoint\-RCNN](#keypoint-rcnn)
  * [3D](#3D)

# Model Zoo

> All experiments are conducted on servers with 8 NVIDIA V100 / 2080Ti GPUs (PCIE). The software in use were PyTorch 1.3, CUDA 10.1, cuDNN 7.6.3.

## ImageNet Classification

Comming Soon.

## Self\-supervised Learning

Comming Soon.

## Object Detection 

### COCO

#### Faster R-CNN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------------------------------------------------------------ |
| [FasterRCNN-R50-FPN](detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.225(2080ti)       | 2.82           | 38.1   | [LINK](detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x/model_final.pth) |
| [FasterRCNN-R50-FPN-SyncBN](detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.2x.syncbn) | 640-800    | 180k     | 0.546               | 5.23           | 39.9   | [LINK](detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.2x.syncbn/model_final.pth) |
| [FasterRCNN-ResNeSt50-FPN](detection/coco/rcnn/faster_rcnn.resnest50.fpn.coco.800size.1x) | 800    | 90k      | 0.416               | 3.53           | 39.9   | [LINK](detection/coco/rcnn/faster_rcnn.resnest50.fpn.coco.800size.1x/model_final.pth) |
| [FasterRCNN-ResNeSt50-FPN-SyncBN](detection/coco/rcnn/faster_rcnn.resnest50.fpn.coco.multiscale.1x.syncbn.4conv) | 640-800    | 90k      | 0.661               | 5.35           | 42.5   | [LINK](detection/coco/rcnn/faster_rcnn.resnest50.fpn.coco.multiscale.1x.syncbn.4conv/model_final.pth) |
| [FasterRCNN-MOBILENET-FPN](detection/coco/rcnn/faster_rcnn.mobilenet.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.279(2080ti)       | 3.47           | 29.27   | [LINK](detection/coco/rcnn/faster_rcnn.mobilenet.fpn.coco.multiscale.1x/model_final.pth) |
| [FasterRCNN-MOBILENET-FPN-NoP2](detection/coco/rcnn/faster_rcnn.mobilenet.fpn.coco.multiscale.1x.no_p2) | 640-800    | 90k      | 0.227(2080ti)       | 2.49           | 29.57   | [LINK](detection/coco/rcnn/faster_rcnn.mobilenet.fpn.coco.multiscale.1x.no_p2/model_final.pth) |

#### RetinaNet

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [RetinaNet-R50](detection/coco/retinanet/retinanet.res50.fpn.coco.800size.1x) | 800    | 90k      | 0.3593               | 3.85           | 35.9   | [LINK](detection/coco/retinanet/retinanet.res50.fpn.coco.800size.1x/model_final.pth) |
| [RetinaNet-R50](detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.244               | 3.84           | 36.5   | [LINK](detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x/model_final.pth) |
| [RetinaNet-R50](detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x.l1_loss) | 640-800    | 90k      | 0.344(2080ti)       | 3.96           | 37.2   | [LINK](detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x.l1_loss/model_final.pth) |
| [RetinaNet-R50-DRLoss](detection/coco/retinanet/retinanet.res50.fpn.coco.800size.1x.dr_loss) | 800        | 90k      | 0.357(2080ti)       | 3.72           | 37.4   | [LINK](detection/coco/retinanet/retinanet.res50.fpn.coco.800size.1x.dr_loss/model_final.pth) |
| [RetinaNet-MOBILENET](detection/coco/retinanet/retinanet.mobilenet.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.266               | 4.46           | 28.5   | [LINK](detection/coco/retinanet/retinanet.mobilenet.fpn.coco.multiscale.1x/model_final.pth) |

#### FCOS

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [FCOS-R50-FPN](detection/coco/fcos/fcos.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.334(2080ti)       | 3.09           | 38.8   | [LINK](detection/coco/fcos/fcos.res50.fpn.coco.800size.1x/model_final.pth) |


#### ATSS

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [ATSS-R50-FPN](detection/coco/atss/atss.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.340(2080ti)       | 3.09           | 39.3   | [LINK](detection/coco/atss/atss.res50.fpn.coco.800size.1x/model_final.pth) |

#### FreeAnchor

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [FreeAnchor-R50-FPN](detection/coco/free_anchor/free_anchor.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.353(2080ti)       | 4.08           | 38.3   | [LINK](detection/coco/free_anchor/free_anchor.res50.fpn.coco.800size.1x/model_final.pth) |

#### TridentNet

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [TridentNet-R50-C4](detection/coco/tridentnet/tridentnet.res50.C4.coco.800size.1x) | 800        | 90k      | 0.754(2080ti)       | 4.65           | 37.7   | [LINK](detection/coco/tridentnet/tridentnet.res50.C4.coco.800size.1x/model_final.pth) |

#### RepPoints

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [RepPoints-R50-FPN](detection/coco/reppoints/reppoints.res50.fpn.coco.800size.1x.partial_minmax) | 800        | 90k      | 0.415(2080ti)       | 2.85           | 38.2   | [LINK](detection/coco/reppoints/reppoints.res50.fpn.coco.800size.1x.partial_minmax/model_final.pth) |

#### CenterNet

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [CenterNet-R18](detection/coco/centernet/centernet.res18.coco.512size) | 512        | 126k     | TBD                 | TBD            | 29.8   | <details><summary>TBD</summary>) |
| [CenterNet-R50](detection/coco/centernet/centernet.res50.coco.512size) | 512        | 126k     | TBD                 | TBD            | 34.9   | <details><summary>TBD</summary>) |
| [CenterNet-R101](detection/coco/centernet/centernet.res101.coco.512size) | 512        | 126k     | TBD                 | TBD            | 36.8   | <details><summary>TBD</summary>) |

#### EfficientDet

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [EffDet0-Effnet0-BiFPN](detection/coco/efficientdet/effdet0.effnet0.bifpn.coco.512size.300e) | 512        | 562k     | 0.540(2080ti)       | 5.77           | 32.6   | [LINK](detection/coco/efficientdet/effdet0.effnet0.bifpn.coco.512size.300e/model_final.pth) |
| [EffDet0-Effnet0-BiFPN-SyncBN](detection/coco/efficientdet/effdet0.effnet0.bifpn.coco.512size.300e.syncbn) | 512        | 562k     | 0.760(2080ti)       | 5.77           | 33.2   | [LINK](detection/coco/efficientdet/effdet0.effnet0.bifpn.coco.512size.300e.syncbn/model_final.pth) |
| [EffDet1-Effnet1-BiFPN](detection/coco/efficientdet/effdet1.effnet1.bifpn.coco.640size.300e) | 640        | 562k     | 0.782(v100)         | 23.18          | 38.1   | [LINK](detection/coco/efficientdet/effdet1.effnet1.bifpn.coco.640size.300e/model_final.pth) |
| [EffDet1-Effnet1-BiFPN-SyncBN](detection/coco/efficientdet/effdet1.effnet1.bifpn.coco.640size.300e.syncbn) | 640        | 562k     | 1.182(v100)         | 23.18          | 38.0   | [LINK](detection/coco/efficientdet/effdet1.effnet1.bifpn.coco.640size.300e.syncbn/model_final.pth) |

#### YOLO

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [YOLOv3-Darknet53-SyncBN](detection/coco/yolo/yolov3.darknet53.coco.multiscale.syncbn) | 320-608    | 470k     | 0.729                 | 7.45            | 37.5   | [LINK](detection/coco/yolo/yolov3.darknet53.coco.multiscale.syncbn/model_final.pth) |

#### SSD

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------------------------------------------------------------ |
| [SSD-VGG16](detection/coco/ssd/ssd.vgg16.coco.300size) | 300        | 200k     | 0.442               | 1.93           | 23.6   | [LINK](detection/coco/ssd/ssd.vgg16.coco.300size/model_final.pth) |
| [SSD-VGG16-Expand](detection/coco/ssd/ssd.vgg16.coco.300size.expand_aug) | 300        | 200k     | 0.448               | 1.93           | 24.9   | [LINK](detection/coco/ssd/ssd.vgg16.coco.300size.expand_aug/model_final.pth) |
| [SSD-VGG16](detection/coco/ssd/ssd.vgg16.coco.512size) | 512        | 200k     | 0.487               | 4.37           | 26.7   | [LINK](detection/coco/ssd/ssd.vgg16.coco.512size/model_final.pth) |
| [SSD-VGG16-Expand](detection/coco/ssd/ssd.vgg16.coco.512size.expand_aug) | 512        | 200k     | 0.491               | 4.37           | 29.0   | [LINK](detection/coco/ssd/ssd.vgg16.coco.512size.expand_aug/model_final.pth) |

#### DETR

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [DETR-R50-C5](detection/coco/detr/detr.res50.c5.coco.multiscale.150e.bs16) | 480-800        | 150e      | 0.270(v100)       | 3.62           | 38.7   | [LINK](detection/coco/detr/detr.res50.c5.coco.multiscale.150e.bs16/model_final.pth) |

#### Sparse R-CNN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [SparseRCNN-R50-FPN](detection/coco/sparse_rcnn/sparse_rcnn.res50.fpn.coco.multiscale.3x) | 480-800        | 270k      | 0.627(2080ti)       | 4.11           | 43.2   | [LINK](detection/coco/sparse_rcnn/sparse_rcnn.res50.fpn.coco.multiscale.3x/model_final.pth) |



### PASCAL VOC

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | AP   | AP50 | AP75 | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ---- | ---- | ---- | ------------------------------------------------------------ |
| [FasterRCNN-R50-FPN](detection/voc/rcnn/faster_rcnn.res50.fpn.voc.multiscale.1x) | 480-800    | 18k      | 0.377               | 2.82           | 54.2 | 82.1 | 59.3 | [LINK](detection/voc/rcnn/faster_rcnn.res50.fpn.voc.multiscale.1x/model_final.pth) |



### WIDER FACE

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ----------------------------------------- |
| [RetinaNet-R50](detection/widerface/retinanet/retinanet.res50.fpn.widerface.600size.0.5x_crop) | 600        | 45k      | 0.342               | 4.76           | 49.4   | [LINK](detection/widerface/retinanet/retinanet.res50.fpn.widerface.600size.0.5x_crop/model_final.pth) |
| [FCOS-R50-FPN](detection/widerface/fcos/fcos.res50.fpn.widerface.600size.0.5x_crop.plus.norm_sync) | 600        | 45k      | 0.382               | 5.75           | 50.8   | [LINK](detection/widerface/fcos/fcos.res50.fpn.widerface.600size.0.5x_crop.plus.norm_sync/model_final.pth) |



### CityPersons

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | MR   | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ---- | ----------------------------------------- |
| [FasterRCNN-R50-FPN](detection/citypersons/rcnn/faster_rcnn.res50.fpn.citypersons.640size.1x) | 640        | 9K       | 0.401               | 3.38           | 36.1   | 0.37 | [LINK](detection/citypersons/rcnn/faster_rcnn.res50.fpn.citypersons.640size.1x/model_final.pth) |
| [RetinaNet-R50](detection/citypersons/retinanet/retinanet.res50.fpn.citypersons.640size.1x) | 640        | 18k      | 0.349               | 2.97           | 33.6   | 0.42 | [LINK](detection/citypersons/retinanet/retinanet.res50.fpn.citypersons.640size.1x/model_final.pth) |
| [FCOS-R50-FPN](detection/citypersons/fcos/fcos.res50.fpn.citypersons.640size.1x) | 640        | 9K       | 0.375               | 3.55           | 35.7   | 0.40 | [LINK](detection/citypersons/fcos/fcos.res50.fpn.citypersons.640size.1x/model_final.pth) |



### CrowdHuman

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | MR   | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ---- | ----------------------------------------- |
| [FasterRCNN-R50-FPN](detection/crowdhuman/rcnn/faster_rcnn.res50.fpn.crowdhuman.800size.1x) | 800        | 2.8K       | 0.856               | 4.80           | 84.1   | 0.481 | [LINK](detection/crowdhuman/rcnn/faster_rcnn.res50.fpn.crowdhuman.800size.1x/model_final.pth) |


## Instance Segmentation 

### COCO

#### Mask R-CNN

| Name                         | input size | lr sched  | train time (s/iter) | train mem (GB)  | box AP | mask AP | Trained Model |
| ---------- | --------- | --------------- | --------------- | ------ | ------- | ---- | ---- |
| [MaskRCNN-R50-C4](segmentation/coco/rcnn/mask_rcnn.res50.c4.coco.multiscale.1x) |  640-800   |    90k    |   0.609         |      5.04       |  36.8  |   32.2  | [LINK](segmentation/coco/rcnn/mask_rcnn.res50.c4.coco.multiscale.1x/model_final.pth)  |
| [MaskRCNN-R50-C4-SyncBN-ExtraNorm](segmentation/coco/rcnn/mask_rcnn.res50.c4.coco.multiscale.1x.syncbn.extra_norm) |  640-800   |    90k    |   0.852         |      9.82       |  37.9  |   33.1  | [LINK](TBD)  |
| [MaskRCNN-R50-C4-SyncBN](segmentation/coco/rcnn/mask_rcnn.res50.c4.coco.multiscale.2x.syncbn) | 640-800 | 180k |   0.837         |      9.82       |  39.9  |   34.5  | [LINK](TBD))  |
| [MaskRCNN-R50-C4-SyncBN-ExtraNorm](segmentation/coco/rcnn/mask_rcnn.res50.c4.coco.multiscale.2x.syncbn.extra_norm) |  640-800   | 180k  |   0.853   |  9.82   |  40.1  |   34.7  | [LINK](segmentation/coco/rcnn/mask_rcnn.res50.c4.coco.multiscale.2x.syncbn.extra_norm/model_final.pth)  |
| [MaskRCNN-R50-FPN](segmentation/coco/rcnn/mask_rcnn.res50.fpn.coco.multiscale.1x) | 640-800 | 90k | 0.297(2080ti) | 3.36 | 38.5 | 35.2 | [LINK](segmentation/coco/rcnn/mask_rcnn.res50.fpn.coco.multiscale.1x/model_final.pth) |

#### TensorMask

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ----------------------------------------- |
| [TensorMask-R50-FPN](segmentation/coco/tensormask/tensormask.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.788(2080ti)       | 7.83           | 37.5   | 32.3    | [LINK](segmentation/coco/tensormask/tensormask.res50.fpn.coco.800size.1x/model_final.pth) |

#### CascadeRCNN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ------------------------------------------------------------ |
| [CascadeRCNN-R50-FPN](segmentation/coco/rcnn/cascade_rcnn.res50.fpn.coco.800size.1x) | 800        | 90k      | 0.546               | 3.91           | 41.7   | 36.1    | [LINK](segmentation/coco/rcnn/cascade_rcnn.res50.fpn.coco.800size.1x/model_final.pth) |

#### PointRend

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ------------------------------------------------------------ |
| [PointRend-R50-FPN](segmentation/coco/pointrend/pointrend.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.439               | 4.88           | 38.4   | 36.2    | [LINK](segmentation/coco/pointrend/pointrend.res50.fpn.coco.multiscale.1x/model_final.pth) |
| [PointRend-R50-FPN](segmentation/coco/pointrend/pointrend.res50.fpn.coco.multiscale.3x) | 640-800    | 270k     | 0.416               | 4.88           | 41.1   | 38.2    | [LINK](segmentation/coco/pointrend/pointrend.res50.fpn.coco.multiscale.3x/model_final.pth) |

#### SOLO

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ------------------------------------------------------------ |
| [SOLO-R50-FPN](segmentation/coco/solo/solo.res50.fpn.coco.800size.1x) | 800    | 90k      | 0.970               | 6.99           | 33.1   | 32.7    | [LINK](segmentation/coco/solo/solo.res50.fpn.coco.800size.1x/model_final.pth) |
| [SOLO-R50-FPN](segmentation/coco/solo/solo.res50.fpn.coco.multiscale.3x) | 640-800    | 270k      | 0.950               | 6.99           | 35.6   | 35.2    | [LINK](segmentation/coco/solo/solo.res50.fpn.coco.multiscale.3x/model_final.pth) |
| [DecoupledSOLO-R50-FPN](segmentation/coco/solo/decoupled_solo.res50.fpn.coco.800size.1x) | 800    | 90k      | 1.097               | 6.68           | 34.0   | 33.7    | [LINK](segmentation/coco/solo/decoupled_solo.res50.fpn.coco.800size.1x/model_final.pth) |
| [DecoupledSOLO-R50-FPN](segmentation/coco/solo/decoupled_solo.res50.fpn.coco.multiscale.3x) | 640-800    | 270k      | 0.922               | 6.47           | 35.9   | 35.6    | [LINK](segmentation/coco/solo/decoupled_solo.res50.fpn.coco.multiscale.3x/model_final.pth) |


### LVIS

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | box AP | mask AP | Trained Model                             |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ | ------- | ----------------------------------------- |
| [MaskRCNN-R50-FPN](segmentation/lvis/rcnn/mask_rcnn.res50.fpn.lvis.800size.1x) | 800        | 90k      | 0.486               | 5.26           | 20.3   | 21.0    | [LINK](segmentation/lvis/rcnn/mask_rcnn.res50.fpn.lvis.800size.1x/model_final.pth) |
| [MaskRCNN-R50-FPN-DataResampling](segmentation/lvis/rcnn/mask_rcnn.res50.fpn.lvis.800size.1x.data_resampling) | 800        | 90k      | 0.500               | 5.26           | 23.0   | 23.1    | [LINK](segmentation/lvis/rcnn/mask_rcnn.res50.fpn.lvis.800size.1x.data_resampling/model_final.pth) |
| [MaskRCNN-R50-FPN-DataResampling](segmentation/lvis/rcnn/mask_rcnn.res50.fpn.lvis.multiscale.1x.data_resampling) | 640-800    | 90k      | 0.485               | 5.25           | 24.1   | 24.7    | [LINK](segmentation/lvis/rcnn/mask_rcnn.res50.fpn.lvis.multiscale.1x.data_resampling/model_final.pth) |



### CITYSCAPES

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | mask AP |                                                              |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------- | ------------------------------------------------------------ |
| [MaskRCNN-R50-FPN](segmentation/cityscapes/rcnn/mask_rcnn.res50.fpn.cityscapes.multiscales.1x) | 640-800    | 90k      | 0.737               | 5.21           | 37.4    | [LINK](segmentation/cityscapes/rcnn/mask_rcnn.res50.fpn.cityscapes.multiscales.1x/model_final.pth) |
| [PointRend-R50-FPN](segmentation/cityscapes/pointrend/pointrend.res50.fpn.cityscapes.multiscale.1x) | 800-1024 | 240k   | 0.746         | 8.21       | 36.0 | [LINK](segmentation/cityscapes/pointrend/pointrend.res50.fpn.cityscapes.multiscale.1x/model_final.pth) |


## Semantic Segmentation

### COCO

#### SemanticFPN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | mIoU | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ---- | ------------------------------------------------------------ |
| [SemanticFPN-R50-FPN](semantic_segmentation/coco/semanticfpn/semanticfpn.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.285               | 6.16            | 40.3 | [LINK](semantic_segmentation/coco/semanticfpn/semanticfpn.res50.fpn.coco.multiscale.1x/model_final.pth) |

### CITYSCAPES

#### PointRend

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | mIoU | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ---- | ------------------------------------------------------------ |
| [PointRend-R101-FPN](semantic_segmentation/cityscapes/pointrend/pointrend.res101.fpn.cityscapes.multiscale.1x) | 512-2048 | 65k    | 1.900             | 3.88        | 78.2 | [LINK](semantic_segmentation/cityscapes/pointrend/pointrend.res101.fpn.cityscapes.multiscale.1x/model_final.pth) |

#### DynamicRouting

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | mIoU | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ---- | ------------------------------------------------------------ |
| [Dynamic-A](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16.SmallGate.Dynamic_A) | 512-2048 | 190k    | 0.736             | 8.74        | 75.7 | [LINK](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16.SmallGate.Dynamic_A/model_final.pth) |
| [Dynamic-B](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16.SmallGate.Dynamic_B) | 512-2048 | 190k    | 0.706             | 8.74        | 75.3 | [LINK](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16.SmallGate.Dynamic_B/model_final.pth) |
| [Dynamic-C](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16.SmallGate.Dynamic_C) | 512-2048 | 190k    | 0.717             | 8.74        | 76.2 | [LINK](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16.SmallGate.Dynamic_C/model_final.pth) |
| [Dynamic-Raw](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16) | 512-2048 | 190k    | 0.757             | 8.73        | 76.5 | [LINK](semantic_segmentation/cityscapes/dynamic_routing/Seg.Layer16/model_final.pth) |

#### FCN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | mIoU | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ---- | ------------------------------------------------------------ |
| [FCN-Res101-s32](semantic_segmentation/cityscapes/fcn/fcn.res101.cityscapes.multiscale.1x.s32) | 512-2048 | 65k    | 0.605             | 3.44        | 71.9 | [LINK](semantic_segmentation/cityscapes/fcn/fcn.res101.cityscapes.multiscale.1x.s32/model_final.pth) |
| [FCN-Res101-s16](semantic_segmentation/cityscapes/fcn/fcn.res101.cityscapes.multiscale.1x.s16) | 512-2048 | 65k    | 0.593             | 3.41        | 73.5 | [LINK](semantic_segmentation/cityscapes/fcn/fcn.res101.cityscapes.multiscale.1x.s16/model_final.pth) |
| [FCN-Res101-s8](semantic_segmentation/cityscapes/fcn/fcn.res101.cityscapes.multiscale.1x.s8) | 512-2048 | 65k    | 0.541             | 3.41        | 74.0 | [LINK](semantic_segmentation/cityscapes/fcn/fcn.res101.cityscapes.multiscale.1x.s8/model_final.pth) |

## Panoptic Segmentation

### COCO

#### PanopticFPN

| Name                                                         | input size | lr sched | train time (s/iter) | train mem (GB) | PG | Trained Model                                                |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ---- | ------------------------------------------------------------ |
| [PanopticFPN-R50-FPN-800](panoptic_segmentation/coco/panoptic_fpn.res50.fpn.coco.800size.1x) | 800    | 90k      | 0.4842               | 4.74            | 39.4 | [LINK](panoptic_segmentation/coco/panoptic_fpn.res50.fpn.coco.800size.1x/model_final.pth) |
| [PanopticFPN-R50-FPN-MS](panoptic_segmentation/coco/panoptic_fpn.res50.fpn.coco.multiscale.1x) | 640-800    | 90k      | 0.4657               | 4.74            | 39.5 | [LINK](panoptic_segmentation/coco/panoptic_fpn.res50.fpn.coco.multiscale.1x/model_final.pth) |

# Key-Points

## COCO_PERSON

### Keypoint-RCNN

| Named                                                        | input size | lr sched | train time (s/iter) | train mem (GB) |  box AP |  kp AP |
| ------------------------------------------------------------ | ---------- | -------- | ------------------- | -------------- | ------ |  ------ |
| [RCNN_R50_FPN](playground/keypoints/coco_person/rcnn/keypoint_rcnn.res50.FPN.coco_person.multiscale.1x) | 480-800    | 90k     | 0.4r0(2080Ti)       | 4.47      | 53.7 | 64.2 | 

# 3D

Comming Soon.
