# TSD

**News**: We reimplement the TSD algorithm based on the MMDetection [framework](https://github.com/open-mmlab/mmdetection).

Paper: 

  TSD (https://arxiv.org/abs/2003.07540)

  1st place solutions for openimage 2019 (https://arxiv.org/abs/2003.07557)

## Introduction

The installation of MMDetection can be found from the official github(https://github.com/open-mmlab/mmdetection)

TSD is a plugin detector head which is friendly to any anchor-based two stage detectors (Faster RCNN, Mask RCNN and so on).

![Overview](demo/TSD.png)

## Changelog

**V1.0**: 
We firstly reimplement the experiments based on Faster RCNN with Resnet families.

The SharedFCBBoxHead is used as the sibling head.

The corresponding configuration can be found in (faster_rcnn_r50_fpn_TSD_1x.py, faster_rcnn_r101_fpn_TSD_1x.py, faster_rcnn_r152_fpn_TSD_1x.py)

### Tips:

1. LR can be set to base_lr\*total_batch (base_lr=0.00125, 0.04 = 0.00125\*32 in our experiments.)
2. An external epoch can be used to perform warmup. (base_lr will be incresed to LR in the first epoch)

## Experiments

Reimplemented methods and backbones are shown in the below table. It's based on the Faster RCNN with FPN.
More backbones and experiments are underway.

| Backbone           | TSD   | AP             | AP_0.5  | AP_0.75  | AP_s    | AP_m      | AP_l     | Download |
|:--------------------:|:-----:|:--------------:|:-------:|:--------:|:-------:|:---------:|:--------:|:--------:|
| ResNet50           |       | 36.2           | 58.1    | 39.0     | 21.8    | 39.9      |46.1      |  |
| ResNet50           | ✓     | **40.9**      | **61.9** | **44.4** |**24.2**  |**44.4**  |**54.0**   |[model](https://drive.google.com/file/d/1G0ngN4Ro5PpcB7S__09Cz3EkAfsWWPy_/view?usp=sharing) |
| ResNet101          |       | 38.9           | 60.6    | 42.4     | 22.3    | 43.6      |50.6      |  |
| ResNet101          | ✓     | **42.3**      | **63.1**| **45.9**  | **25.1**|**46.3**  |**56.5**    |[model](https://drive.google.com/open?id=1FghatPmrWx8QPeZaOn-dODJP3nqu9Jdj) |
| ResNet152          |       |  40.5        |62.1      |44.5     | 24.6     |45.0       | 51.8      | |
| ResNet152          | ✓     | **43.7**     |**64.5**  |**47.6** |**26.1**  |**48.0**   |**57.5**   |[model](https://drive.google.com/open?id=1OQTkZIzNZ323BBxsxwMbl6YDYAgAfvb0)|

### TBD

**We will continue to update the pretrained models of some heavy backbones.**

## Installation

Please refer to [MMdetection](docs/INSTALL.md) for installation and dataset preparation.


## Get Started
```shell
./tools/slurm_train.sh dev TSD configs/faster_rcnn_r152_fpn_TSD_1x.py exp/TSD_r152/ 16
```

## Acknowledgement

We sinerely appreciate the support of MMDetection for object detection algorithms.

## Citations

If the TSD helps your research, please cite the follow papers.

```
@article{song2020revisiting,
  title={Revisiting the Sibling Head in Object Detector},
  author={Song, Guanglu and Liu, Yu and Wang, Xiaogang},
  journal={arXiv preprint arXiv:2003.07540},
  year={2020}
}
@article{liu20201st,
  title={1st Place Solutions for OpenImage2019--Object Detection and Instance Segmentation},
  author={Liu, Yu and Song, Guanglu and Zang, Yuhang and Gao, Yan and Xie, Enze and Yan, Junjie and Loy, Chen Change and Wang, Xiaogang},
  journal={arXiv preprint arXiv:2003.07557},
  year={2020}
}
```


## Contact

If you have any questions, please contact (songguanglu@sensetime.com).
