---
title: Deep High-Resolution Representation Learning for Human Pose Estimation
date: 2019-05-23 14:29:55
description: CVPR-2019 Deep High-Resolution Representation Learning for Human Pose Estimation
categories: Deep Learning
tags:
- Deep Learning
- Pose Estimation
---

[Paper Link](https://arxiv.org/abs/1902.09212)

[Code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

# 介绍

High-Resolution Net (HRNet)

这篇论文解决人体姿态估计问题，重点关注学习高分辨率表示。大多数现有的方法通过一个由高到低分辨率的网络，从一个低分辨率的表示中恢复高分辨率的表示。相反，本论文提出的网络自始至终都保持了高分辨率的表示。

作者从高分辨率子网作为第一个阶段，逐渐添加 高->低 分辨率的子网形成更多的阶段，并且并行连接这些多分辨率子网。作者多次进行多尺度融合，使得每一个高->低分辨率的表示可以从其他并行的表示中接收信息，从而生成丰富的高分辨率表示。因此，预测的关键点热图可以更准确，空间更精确。作者在COCO关键点检测数据集和MPII Human Pose数据集上进行了验证。



# 与其他方法之前的区别

1. 该方法并行级联高->低分辨率子网，而不是以序列的方式。因此，该方法保持了高分辨率，而不是从低分辨率中恢复高分辨率。所以预测的热图的空间上更准确。

2. 现存的融合策略集成底层（low-level）和高层（high-level）的表示。 而该方法通过相似深度和相同层级的低分辨率表示的帮助，执行重复的多尺度融合提升高分率表示。

{% asset_img cVPR19-HRNet.png Figure 1. 提出的HRNet的结构%}

图1展示了提出的HRNet网络的结构。它包含并行的高->低分辨率的子网，重复的在不同分辨率子网之间的信息交换，即多尺度融合。水平和垂直方向分别对应于网络的深度和特征图的比例。


{% asset_img hRNet-framework.png Figure 2. 依靠high-to-low和low-to-high框架的姿态估计网络结构 %}

图2展示了其他方法的一些网络结构，这些方法都是依靠high-to-low和low-to-high框架的姿态估计网络结构。其中（a)表示Hourglass网络，（b）表示Cascaded pyramid networks，（c）表示SimpleBaseline网络：转置卷积（transposed convolutions）用于low-to-high过程。（d）组合空洞卷积（dilated convolutions）。


# 方法介绍

## 序列多尺度子网

用$N_{sr}$表示子网络在第s个stage，r表示分辨率的序号，它的分辨率是第一个子网络分辨率的$\frac{1}{r^{r-1}}$倍。有S=4个stege的high-to-low网络可以表示为：

$$N_{11} \to N_{22} \to N_{33} \to N_{44}$$

## 并行多尺度子网

我们从一个高分辨率子网络作为第一个stage起始，逐渐地增加high-to-low分辨率子网络，形成新的sgates，并且并行地连接多分辨率子网络。因此，后一阶段并行子网的分辨率包括前一阶段的分辨率和一个更低的分辨率。

这里给出一个网络结构的例子，包含4个并行的子网络，如下：

{% asset_img hRNet-eq2.png%}

## 重复多尺度融合

{% asset_img hRNet-exchange-unit.png Figure 3. Exchange Unit%}

图3展示了交换单元（Exchange Unit）如何为高、中和底层融合信息的。右侧的注释表示：strided 3x3=stride 3x3卷积，up samp. 1x1=最近邻上采样和一个1x1卷积。

我们在不同的并行子网之间引入交换单元（exchange unit），这样每个子网可以重叠地从其他并行网络中接收信息。这里给出了一个交换信息框架的例子，如下图表示的结构。我们将第三个stage分成几个exchange blocks，并且每一个block有三个并行的卷积单元构成，一个交换单元在并行的卷积单元之间，如下：

{% asset_img hRNet-eq3.png%}

其中，$C^b_{sr}$表示在第s个stage，第b个block的第r分辨率的卷积单元。$\varepsilon^b_s$是对应的交换的单元。

交换单元如图3所示。

{% asset_img hRNet-exchange-unit-2.png%}


## 热图估计

我们简单地从最后一个交换单元（exhcange unit）输出的高分辨率表示中回归热图。损失函数（定义为均方误差）用于比较预测的热图和groundtruth热图。通过应用2D高斯生成的groundtruth热图，其中标准偏差为1像素，并以每个关键点的标注位置为中心。

## 网络实例

实验中提出了两种网络，一个小网络HRNet-W32，一个大网络HRNet-W48，其中32和48分别表示在后3个sgate中的高分辨率子网络的宽度（C）。对于HRNet-W32，其他三个并行的子网络的宽度分别是64，128,256，对于HRNet-W48是96，192，384。



# 实验



{% asset_img hRNet-COCO-validation.png Figure 1. %}

{% asset_img hRNet-COCO-test.png Figure 1. %}

{% asset_img hRNet-MPII.png Figure 1.%}

{% asset_img hRNet-Qualitative-Results.png Figure 1.%}

{% asset_img hRNet-1x2x4x.png Figure 1.%}

{% asset_img hRNet-SimpleBaseline-performance.png Figure 1.%}

# 代码

## Exchange Unit

```python
    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)
```

## HighResolutionModule

```python
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse
```
