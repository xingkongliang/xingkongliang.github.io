---
title: Panoptic Feature Pyramid Networks
date: 2019-01-11 11:10:04
description:
categories:
tags:
---
论文链接：http://arxiv.org/abs/1901.02446

## Abstract

全景分割(panoptic segmentation)

实例分割(instance segmentation)(for thing classes)

语义分割(semantic segmentation)(for stuff classes)

之前的一些方法使用了分开的不相似的网络做实例分割和语义分割，它们之间没有任何共享计算。在这个工作中，作者在结构层面统一了这些方法，设计了一个单一网络用于这些任务。

该方法使用一个语义分割分支扩展Mask-RCNN，使其具有全景分割的能力。该分支使用了一个共享的特征金字塔网络主干。该方法不仅保持了实例分割的性能，也得到了一个轻量级权重的、高性能的语义分割方法。

<!--more-->

---

{% asset_img panopticFPN.png Figure 1. Panoptic FPN. %}


{% asset_img panopticfpnResults.png Figure 2. Panoptic FPN results on COCO (top) and Cityscapes (bottom) using a single ResNet-101-FPN network. %}

图２展示了Panoptic　Feature　Pyramid　Networks的预测结果。





## Panoptic Feature Pyramid Network
该网络结构在原有的Feature Pyramid Network的基础上，保持了原有的Instance segmentation branch，新提出了Panoptic FPN。

Panoptic FPN: As discussed, our approach is to modify Mask R-CNN with FPN to enable pixel-wise semantic seg- mentation prediction. However, to achieve accurate predic- tions, the features used for this task should: (1) be of suit- ably high resolution to capture fine structures, (2) encode sufficiently rich semantics to accurately predict class labels, and (3) capture multi-scale information to predict stuff re- gions at multiple resolutions. Although FPN was designed for object detection, these requirements – high-resolution, rich, multi-scale features – identify exactly the characteris- tics of FPN. We thus propose to attach to FPN a simple and fast semantic segmentation branch, described next.

### Semantic segmentation branch

{% asset_img semanticSegmentationBranch.png Figure 3: Semantic segmentation branch. %}
图３显示了语义分割分支。左侧的每个FPN层通过卷积和双线性插值上采样被上采样，直到它达到1/4的尺度（右侧），这些输出被加和，并且最终转换为一个像素级别的输出。

重点介绍语义分割分支：

为了从FPN特征生成语义分割的输出，作者提出一个简单的设计去融合来自FPN金字塔层中所有信息到一个单一的输出，如上图３所示。开始与最深层的FPN(在尺度1/32上)，我们执行山歌上采样阶段去产生一个1/4尺度的特征图，这里每一个上采样阶段包含3x3卷积，group norm, ReLU，和2x bilinear upsampling。这些策略被重复用在其他FPN尺度上(1/16,1/8和1/4)，并且渐进地减少上采样步骤。这个结果是在1/4尺度的输出特征层的一个集合，该集合是通过元素相加的方法组合成的。1x1卷积，4x bilinear upsampling和softmax被用于在原始图像分辨率上生成每个像素的类别标签。


{% asset_img semanticSegmentationBranchTmp.png%}

{% asset_img backboneArchitecturesPFPN.png Figure 5: Backbone architectures for increasing feature resolution. %}

图５是用于增加分辨率的骨干网结构。(a)是标准的卷积网络，(维度定义为#blocksx#channels#xresolution)。(b)通过使用空洞卷积(dilated convolutions)来减少卷积的步长。(c)一个U-Net风格的网络，使用一个对称的解码器镜像bottom-up通路，但是是反向的。(d)FPN可以被视为一个非对称的、轻权重的解码器，它的top-down通路中每个stage仅仅有一个block，并且使用了一个相同的通道维度。
