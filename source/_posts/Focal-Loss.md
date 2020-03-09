---
title: Focal Loss for Dense Object Detection
date: 2019-01-10 21:03:52
description:
categories: Object Detection
tags: Object Detection
---



本文代码：https://github.com/facebookresearch/Detectron

本文主要解决在one-stage的密集检测器中，大量前景和背景不平衡的问题。作者通过降低被很好分类样本的权重来解决类别不平衡的问题。Focal Loss集中于在稀疏难样本(hard examples)上的训练，并且在训练中防止大量的容易的反例(easy negatives)淹没检测器。

1. 提出Focal Loss, 解决正负样本不平衡问题;
2. 提出one-stage检测模型，RetinaNet。

<!--more-->

{% asset_img focalLoss.png Figure 1. Focal Loss %}

作者提出一个新的损失函数 Focal Loss, 该方法通过添加因子$(1-p_t)^\gamma$到标准的交叉熵损失函数。设定$\gamma>0$会减少被很好分类样本($p>0.5$)的相对损失，更加关注于难和错误分类的样本。


{% asset_img seedVs.Accuracy.png Figure 2. Speed (ms) versus accuracy (AP) on COCO test-dev. %}

图中显示了，RetinaNet检测器使用了focal loss，结果由于之前的one-stage和two-stage检测器。

### 类别不平衡问题
在R-CNN这一类检测器中，类别不平衡问题通过two-stage级联和采样策略被解决。在候选区域提取阶段，Selective Search, EdgeBoxes, RPN等方法，缩小候选区域位置的数量到１~2k个，大量过滤掉了背景。在第二分类阶段，采样策略，例如固定前景背景比率(1:3)，或者在线难样本挖掘(OHEM)方法被执行用于保持前景和背景的一个可控的平衡。

## Focal Loss

Focal Loss被设计用以解决one-state目标检测器训练中大量正反例样本不平衡的问题，通常是(1:1000)。我们首先介绍二值分类的交叉熵损失:

$$ CE(p, y)=\begin{cases}
-log(p) & y=1\\
-log(1-p) & otherwise.
\end{cases} $$


### 3.1 Balanced Cross Entropy

一种通用的解决类别不平衡的方法是对类别1引入一个权重因子$\alpha \in [0, 1]$，对类别class-1引入$1-\alpha$。我们写成$\alpha-$balanced CE loss:

$$CE(p_t)=-\alpha_t log(p_t)$$

### 3.2 Focal Loss Definition

试验中显示，在训练dense detectors是遭遇的大量类别不平衡会压倒交叉熵损失。容易分类的样本会占损失的大部分，并且主导梯度。尽管$\alpha$平衡了正负(positive/negative)样本的重要性，但是它没有区分易和难的样本(easy/hard)。相反，作者提出的更改过后的损失函数降低了容易样本的权重并且集中训练难反例样本。

我们定义focal loss：

$$FL(p_t)=-(1-p_t)^\gamma log(p_t)$$

### 3.4 Class Imbalance and Two-stage Detectors

Two-stage检测器通常没有使用$\alpha-$balancing 或者我们提出的loss。代替这些，他们使用了两个机制来解决类别不平衡问题：(1) 一个两级的级联，(2) 有偏置的小批量采样。第一级联阶段的候选区域提取机制减少了大量可能的候选位置。重要的是，这些选择的候选框不是随机的，而是选择更像前景的可能位置，这样就移除了大量的容易的难反例样本(easy negatives)。第二阶段，有偏置的采样通常使用1:3比率的正负样本构建小批量(minibatches)。这个采样率类似$\alpha-$balancing因子，并且通过采样来实现。作者提出的focal loss主要设计用于解决one-stage检测系统中的这些问题。

{% asset_img ablationExperimentsForFocalLoss.png Table 1. Ablation experiments for RetinaNet and Focal Loss (FL). %}

{% asset_img objectDetectionRetinanet.png Table 2. Object detection single-model results (bounding box AP), vs. state-of-the-art on COCO test-dev. We %}
