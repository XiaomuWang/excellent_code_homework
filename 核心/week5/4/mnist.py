#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/20 19:51 

# @Author : ZFJ

# @File : mnist.py 

# @Software: PyCharm
"""

import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from kmeanspp import *

epochNum = 100
batchSize = 100
trainImgNum = batchSize * 1

transform = transforms.ToTensor()

# 训练集
trainSet = tv.datasets.MNIST(
    root='./train',
    train=True,
    download=True,
    transform=transform)

trainLoader = DataLoader(
    trainSet,
    batch_size=batchSize,
    shuffle=True)


# 老师的列投影feature
def get_feature(x):
    feature = [0, 0, 0, 0]

    # 下面添加提取图像x的特征feature的代码
    def get_shadow(x, dim):
        feature = torch.sum(x, dim)
        feature = feature.float()
        # 归一化
        for i in range(0, feature.shape[0]):
            s = sum(feature) + 1e-15
            feature[i] = feature[i] / s

        feature = feature.view(1, 28)
        return feature

    feature = get_shadow(x.reshape((28, 28)), 0)
    return feature


# 随便挑100张图片来训练
trainImgs = []
for batch, (imgs, labels) in enumerate(trainLoader):
    for img in imgs:
        trainImgs.append(get_feature(img))

    if batchSize * (batch + 1) >= trainImgNum:
        break

kmeanspp = Kmeanspp(10, trainImgs)
for epoch in range(0, epochNum):
    print("epoch {}/{}".format(epoch, epochNum))
    kmeanspp.step()

# 随便拿100张图片来测试
for batch, (imgs, labels) in enumerate(trainLoader):
    for i, img in enumerate(imgs):
        clusterType = kmeanspp.getClusterType(get_feature(img))
        print("数字 [{}] 被分到第 [{}] 类".format(labels[i].item(), clusterType))

    if batchSize * (batch + 1) >= trainImgNum:
        break
