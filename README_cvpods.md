<div align=center><img src=".github/cvpods_logo.png" width="400" ><div align=left>

[![cvpods compliant](https://img.shields.io/badge/cvpods-master-brightgreen)](https://github.com/Megvii-BaseDetection/cvpods)
![ci](https://github.com/Megvii-BaseDetection/cvpods/workflows/build/badge.svg?branch=master)

Welcome to **cvpods**, a versatile and efficient codebase for many computer vision tasks: classification, segmentation, detection, self-supervised learning, keypoints and 3D(classification / segmentation / detection / representation learing), etc. The aim of cvpods is to achieve efficient experiments management and smooth tasks-switching.

<div align=center><img src=".github/cvpods_tasks.png"><div align=left>

>  Each sub-image denotes a task. All images are from search engine.

## Table of Contents

- [Changelog](#changelog)
- [Install](#install)
- [Usage](#usage)
	- [Get started](#get-start)
	- [Step-by-step tutorial](#tutorials)
- [Model Zoo](#model-zoo)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Changelog 
* Dec. 03, 2020: cvpods v0.1 released.

## Install

### Requirements

* Linux with Python ≥ 3.6
* PyTorch ≥ 1.3 and torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this
* OpenCV is optional and needed by demo and visualization

### Build cvpods from source 

**Make sure GPU is available on your local machine.**

```shell
# Install cvpods with GPU directly 
pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git' --user

# Or, to install it with GPU from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
pip install -e cvpods --user 

# Or, to build it without GPU from a local clone:
FORCE_CUDA=1 pip install -e cvpods --user

```

## Usage
Here we demonstrate the basic usage of cvpods (Inference & Train). For more features of cvpods, please refer to our documentation or provided tutorials.

### Get Start 
Here we use coco object detection task as an example.
```
# Preprare data path
ln -s /path/to/your/coco/dataset datasets/coco

# Enter a specific experiment dir 
cd playground/retinanet/retinanet.res50.fpn.coco.multiscale.1x

# Train
pods_train --num-gpus 8
# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"
```

### Tutorials

We provide a detailed tutorial, which covers introduction, usage, and extend guides in [cvpods_tutorials](https://github.com/poodarchu/cvpods/blob/master/docs/tutorials/cvpods%20tutorials.ipynb). For all API usages, please
refer to our [documentation](https://cvpods.readthedocs.io/).

## Model ZOO 

For all the models supported by cvpods, please refer to [MODEL_ZOO](https://github.com/Megvii-BaseDetection/cvpods/blob/master/playground/README.md). We provide 50+ methods across ~15 dataset and ~10 computer vision tasks. cvpods has also supported many research projects of MEGVII Research.

### Projects based on cvpods
> List is sorted by names.
* [AutoAssign](https://github.com/Megvii-BaseDetection/AutoAssign)
* [BorderDet](https://github.com/Megvii-BaseDetection/BorderDet)
* [DeFCN](https://github.com/Megvii-BaseDetection/DeFCN)
* [DynamicHead](https://github.com/StevenGrove/DynamicHead)
* [DynamicRouting](https://github.com/Megvii-BaseDetection/DynamicRouting)
* [LearnableTreeFilterV2](https://github.com/StevenGrove/LearnableTreeFilterV2)
* [SelfSup](https://github.com/poodarchu/SelfSup)


## Contributing 
Any kind of contributions (new models / bug report / typo / docs) are welcomed. Please refer to [CONTRIBUTING](CONTRIBUTING.md) for more details.

## License

[Apache v2](LICENSE) © Base Detection 

## Acknowledgement

cvpods adopts many components (e.g. network layers) of Detectron2, while cvpods has many advantanges in task support, speed, usability, etc. For more details about official detectron2, please check [DETECTRON2](https://github.com/facebookresearch/detectron2/blob/master/README.md)
