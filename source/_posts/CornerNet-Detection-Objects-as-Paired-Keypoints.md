---
title: 'CornerNet: Detection Objects as Paired Keypoints'
date: 2018-09-02 22:08:13
description:
categories: Object Detection
tags:
 - Object Detection
---

### 前言

CornerNet: Detection Objects as Paired Keypoints　这篇论文发表在ECCV2018，本人感觉非常有意思，所以和大家分享一下。

Arxiv: https://arxiv.org/abs/1808.01244
Github: https://github.com/umich-vl/

---
### 介绍

传统的目标检测都是给出紧致的候选框，本论文独具匠心，通过一对关键点（目标的左上角和右下角）来检测一个目标框。通过检测关键点的这种方式，可以消除利用先验知识设计anchor boxes这个需求。作者提出角点池化（corner pooling），角点池化可以帮助网络更好的定位角点。最终实验表明，CornerNet在MS COCO数据集上实现了42.1%的AP，优于所有现存的单级(one-stage)检测器。

<!--more-->
---

{% asset_img Fig1.png We detect an object as a pair of bounding box corners grouped together. %}

{% asset_img Fig2.png Often there is no local evidence to determine the location of a bounding box corner. We address this issue by proposing a new type of pooling layer. %}

{% asset_img Fig3.png Corner pooling: for each channel, we take the maximum values (red dots) in two directions (red lines), each from a separate feature map, and add the two maximums together (blue dot). %}

{% asset_img Fig4.png Overview of CornerNet. The backbone network is followed by two prediction modules, one for the top-left corners and the other for the bottom-right corners. Using the predictions from both modules, we locate and group the corners. %}

{% asset_img Fig5.png “Ground-truth” heatmaps for training. %}

{% asset_img Fig6.png The top-left corner pooling layer can be implemented very efficiently. We scan from left to right for the horizontal max-pooling and from bottom to top for the vertical max-pooling. We then add two max-pooled feature maps. %}

{% asset_img Fig7.png The prediction module starts with a modified residual block, in which we replace the first convolution module with our corner pooling module. The modified residual block is then followed by a convolution module. We have multiple branches for predict- ing the heatmaps, embeddings and offsets %}

{% asset_img Fig8.png Example bounding box predictions overlaid on predicted heatmaps of corners. %}
