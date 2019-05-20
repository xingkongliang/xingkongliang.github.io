---
title: Squeeze-and-Excitation Networks
date: 2019-05-17 14:57:27
description: SENet
categories: Deep Learning
tags: Deep Learning
---
# SENet介绍

卷积神经网络（CNNs）的核心模块是卷积操作，这个操作使得网络能够通过每层的局部感受野融合空间和通道的信息，来构建有信息的特征。之前大量的工作已经研究了这种关系的空间组成部分，试图通过提高整个特征层次中空间编码的质量来增强CNN的表征能力。在这项工作中，作者将重点放在通道关系上，并且提出一个新的构架单元，成为“Squeeze-and-Excitation”（SE）块，通过明确地建模通道之间的相互依赖性来自适应地重新校准通道方面的特征响应。

# Squeeze-and-Excitation Blocks
{% asset_img sENet.png %}


{% asset_img sENet-eq1.png %}

## Squeeze: 全局信息嵌入
{% asset_img sENet-eq2.png %}

## Excition: 适应性地校准
{% asset_img sENet-eq3.png %}

{% asset_img sENet-eq4.png %}


# 实例化到ResNet和Inception


{% asset_img sE-Inception.png Figure 1. SE-Inception module %}

{% asset_img sE-ResNet.png Figure 1. SE-Inception module %}

{% asset_img sENet-Table1.png Figure 1. SE-Inception module %}

# 代码

## Caffe

[Caffe SENet](https://github.com/hujie-frank/SENet)

## 第三方实现
0. Caffe. SE-mudolues are integrated with a modificated ResNet-50 using a stride 2 in the 3x3 convolution instead of the first 1x1 convolution which obtains better performance: [Repository](https://github.com/shicai/SENet-Caffe).
0. TensorFlow. SE-modules are integrated with a pre-activation ResNet-50 which follows the setup in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch): [Repository](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet).
0. TensorFlow. Simple Tensorflow implementation of SENets using Cifar10: [Repository](https://github.com/taki0112/SENet-Tensorflow).
0. MatConvNet. All the released SENets are imported into [MatConvNet](https://github.com/vlfeat/matconvnet): [Repository](https://github.com/albanie/mcnSENets).
0. MXNet. SE-modules are integrated with the ResNeXt and more architectures are coming soon: [Repository](https://github.com/bruinxiong/SENet.mxnet).
0. PyTorch. Implementation of SENets by PyTorch: [Repository](https://github.com/moskomule/senet.pytorch).
0. Chainer. Implementation of SENets by Chainer: [Repository](https://github.com/nutszebra/SENets).

## Pytorch实现SE模块

来自https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py的se_module.py文件


```

from torch import nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```
