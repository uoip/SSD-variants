

This is a learning project trying to implement some varants of SSD in pytorch. SSD is a one-stage object detector, probably "currently the best detector with respect to the speed-vs-accuracy trade-off". There are many follow-up papers that either further improve the detection accuracy, or incorporate techniques like image segmentation to be used for Scene Understanding(e.g. [BlitzNet](https://arxiv.org/abs/1708.02813)), or modify SSD to detect rotatable objects(e.g. [DRBox](https://arxiv.org/abs/1711.09405)), or apply SSD to 3d object detection(e.g. [Frustum PointNets](https://arxiv.org/abs/1711.08488)):


* SSD - "SSD: Single Shot MultiBox Detector" (2016) [arXiv:1512.02325](https://arxiv.org/abs/1512.02325) , [github](https://github.com/weiliu89/caffe/tree/ssd)
* DSSD - "DSSD : Deconvolutional Single Shot Detector" (2017) [arXiv:1701.06659](https://arxiv.org/abs/1701.06659)
* RRC - "Accurate Single Stage Detector Using Recurrent Rolling Convolution" (2017) [arXiv:1704.05776](https://arxiv.org/abs/1704.05776) , [github](https://github.com/xiaohaoChen/rrc_detection)
* RUN - "Residual Features and Unified Prediction Network for Single Stage Detection" (2017) [arXiv:1707.05031](https://arxiv.org/abs/1707.05031)
* DSOD - "DSOD: Learning Deeply Supervised Object Detectors from Scratch" (2017) [arXiv:1708.01241](https://arxiv.org/abs/1708.01241) , [github](https://github.com/szq0214/DSOD)
* BlitzNet - "BlitzNet: A Real-Time Deep Network for Scene Understanding" (2017) [arXiv:1708.02813](https://arxiv.org/abs/1708.02813) , [github](https://github.com/dvornikita/blitznet)
* RefineDet - "Single-Shot Refinement Neural Network for Object Detection" (2017) [arXiv:1711.06897](https://arxiv.org/abs/1711.06897) , [github](https://github.com/sfzhang15/RefineDet)
* DRBox - "Learning a Rotation Invariant Detector with Rotatable Bounding Box" (2017) [arXiv:1711.09405](https://arxiv.org/abs/1711.09405) , [github](https://github.com/liulei01/DRBox)
* Frustum PointNets - "Frustum PointNets for 3D Object Detection from RGB-D Data" (2017) [arXiv:1711.08488](https://arxiv.org/abs/1711.08488)

## Overview
| Model      |publish time| Backbone  |  input size  | Boxes | FPS |  VOC07 | VOC12 | COCO |
| -----------|:--------|--------------|--------------|-------|-----|--------|-------|--------|
| SSD300     | 2016    | VGG-16       |  300 × 300   | 8732  | 46  | 77.2   | 75.9  | 25.1
| SSD512     | 2016    | VGG-16       |  512 × 512   | 24564 | 19  | 79.8   | 78.5  | 28.8
| SSD321     | 2017.01 | ResNet-101   |  321 × 321   | 17080 | 11.2| 77.1   | 75.4  | 28.0
| SSD513     | 2017.01 | ResNet-101   |  513 × 513   | 43688 | 6.8 | 80.6   | 79.4  | 31.2
| DSSD321    | 2017.01 | ResNet-101   |  321 × 321   | 17080 | 9.5 | 78.6   | 76.3  | 28.0
| DSSD513    | 2017.01 | ResNet-101   |  513 × 513   | 43688 | 5.5 | 81.5   | 80.0  | 33.2
| RUN300     | 2017.07 | VGG-16       |  300 × 300   | 11640 | 64 (Pascal)  | 79.1  | 77.0  | 
| DSOD300    | 2017.08 |DS/64-192-48-1|  300 × 300   |       | 17.4| 77.7   | 76.3  | 29.3
| BlitzNet300| 2017.08 | ResNet-50    |  300 × 300   | 45390 | 24  | 78.5   | 75.4  | 29.7
| BlitzNet512| 2017.08 | ResNet-50    |  512 × 512   | 32766 | 19.5| 80.7   | 79.0  | 34.1
|RefineDet320| 2017.11 | VGG-16       |  320 × 320   | 6375  | 40.3| 80.0   | 78.1  | 29.4
|RefineDet512| 2017.11 | VGG-16       |  512 × 512   | 16320 | 24.1| 81.8   | 80.0  | 33.0
|RefineDet320| 2017.11 | ResNet-101   |  320 × 320   |       |     |        |   | 32.0
|RefineDet512| 2017.11 | ResNet-101   |  512 × 512   |       |     |        |   | 36.4
| | | | | | | | |
| RRC        | 2017.04 | VGG-16       | 1272 × 375
| DRBox      | 2017.11 | VGG-16       |  300 × 300
| Frustum PointNets rgb part | 2017.11 | VGG-16 | 1280 × 384

* FPS: # of processed images per second  on Titan X GPU (batch size is 1)
* VOC07: PASCAL 2007 detection results(mAP), training data: 07+12(07 trainval + 12 trainval)
* VOC12: PASCAL 2012 detection results(mAP), training data: 07++12(07 trainval + 07 test + 12 trainval)
* COCO: MS COCO 2015 test-dev detection results(mAP@[0.5:0.95]), train on trainval35k

All backbone networks above have been pre-trained on ImageNet CLS-LOC dataset, except DSOD, it's "training from scratch".

  
## Implemented
- [x] SSD
- [x] RRC
- [x] RUN
- [x] DSOD
- [x] BlitzNet (detection part)
- [ ] DRBox
- [ ] Frustum PointNets

**Note:** "Implemented" above means the code of the model is almost done, it doesn't mean I have trained it, or even reproduced the results of original paper. Actually, I have only trained SSD300 on VOC07, the best result I got is 76.5%, lower than 77.2% reported in [SSD paper](https://arxiv.org/abs/1512.02325). I'll continue this project when I find out what's the problem.  

## Requirements
* Python 3.6+
* numpy
* cv2
* pytorch
* tensorboardX

## Dataset
Download dataset VOC2007 and VOC2012, put them under `VOCdevkit` directory:
```
VOCdevkit
-| VOC2007
   -| Annotations
   -| ImageSets
   -| JPEGImages
   -| SegmentationClass
   -| SegmentationObject
-| VOC2012
   -| Annotations
   -| ImageSets
   -| JPEGImages
   -| SegmentationClass
   -| SegmentationObject
```

## Usage
### train:
`python train.py --cuda --voc_root path/to/your/VOCdevkit --backbone path/to/your/vgg16_reducedfc.pth`  
The backbone network `vgg16_reducedfc.pth` is from repo [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) (download link: https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth).

### evaluate:
`python train.py --cuda --test --voc_root path/to/your/VOCdevkit --checkpoint path/to/your/xxx.pth`  

### show demo:
`python train.py --cuda --demo --voc_root path/to/your/VOCdevkit --checkpoint path/to/your/xxx.pth`

## Results
VOC07 mAP
|  models  | my result | paper result |
|----------|-----------|--------------|
|  SSD300  |  76.5%   |    77.2%     |

to be continued

## Reference
* [ChainerCV](https://github.com/chainer/chainercv)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)