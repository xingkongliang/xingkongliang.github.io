---
title: Libra R-CNN
date: 2019-09-27 11:30:17
description: Libra R-CNN Towards Balanced Learning for Object Detection
categories: Object Detection
tags: Object Detection
---

- **[Libra R-CNN]** Libra R-CNN: Balanced Learning for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.02701.pdf) | [[Libra R-CNN 知乎](https://www.zhihu.com/question/319458937/answer/647082241)]

{% asset_img 20190927_Libra_R-CNN_Figure1.png Figure 1 %}


## 1.方法

## 1.1 IoU-balanced Sampling

我们首先提一个问题：一个训练样本和它对应 ground truth 之间的重叠面积与它的困难程度相关么？

{% asset_img 20190927_Libra_R-CNN_Figure2_IoU_distribution_random.png Figure 2 %}


我们主要考虑难反例，因为它是主要问题。是超过 60% 的难样本有大于 0.05 的重叠面积，但是随机采样在大于相同阈值的情况下，只会提供 30% 的训练样本．这种极端的样品失衡将许多硬样品埋入成千上万个简单样品中。这种极端的样本失衡将许多难样本埋入成千上万个简单让本中。

基于这个观察，作者提出了 IoU-balanced sampling．假如我们需要从 M 个候选框中采样 N 个反例样本．在随机采样中，每个样本被选择的概率是：

$$p=\frac{N}{M}.$$

为了提升难反例样本的选择概率，我们根据 IoU 将采样间隔平均分到 K 个尺度（bin）中．N demanded negative samples are equally distributed to each bin.　然后我们从它们中均匀采样．因此，在 IoU-balanced sampling 下，我们得到的样本被选择的概率是：

$$p_k = \frac{N}{K} * \frac{1}{M_k}, k \in [0, K)$$

这里 $M_k$ 是对应第 k 个间隔上采样候选区域的数量．K 在实验中默认设置为 3 ．

从上图中可以看出，作者提出的 IoU-balanced sampling 可以引导训练样本的分布更加接近于难反例样本的分布．只要具有更大的 IoU 的样本尽可能的被选择，实验显示性能对 K 的设置不敏感．

除此之外，这个方法也可以适用于难正例样本．但是，在大多数情况下，没有足够的采样候框将此过程扩展为正例样本．为了使平衡采样程序更加全面，我们对每个 ground truth 都采样了相等的正样本作为替代方法。

## 1.2 Balanced Feature Pyramid

{% asset_img 20190927_Libra_R-CNN_Figure3_balanced_feature_pyramid.png Figure 3 %}


### Obtaining balanced semantic features
$$C=\frac{1}{L} \sum_{l=l_{min}}^{l_{max}} C_l.$$



### Refining balanced semantic features

我们使用了两种精炼方法，分别是 convolutions 和 non-local module．

## 1.3 Balanced L1 Loss
