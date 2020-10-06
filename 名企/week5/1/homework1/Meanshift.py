#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/27 15:16 

# @Author : ZFJ

# @File : Meanshift.py 

# @Software: PyCharm
"""
"""
调用Meanshift实现聚类
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

coordinate, type_index = make_blobs(
    # 1000个样本
    n_samples=1000,
    # 每个样本2个特征,代表x和y
    n_features=2,
    # 4个中心
    centers=4,
    # 随机数种子
    random_state=2
)
fig0, axi0 = plt.subplots(1)
# 传入x、y坐标，macker='o'代表打印一个圈，s=8代表尺寸
axi0.scatter(coordinate[:, 0], coordinate[:, 1], marker='o', s=8)
# 打印所有的点
plt.show()
color = np.array(['red', 'yellow','blue','black'])
fig1, axi1 = plt.subplots(1)
# 下面显示每个点真实的类别
for i in range(4):
    axi1.scatter(
        coordinate[type_index == i, 0],
        coordinate[type_index == i, 1],
        marker='o',
        s=8,
        c=color[i]
    )
plt.show()
# 估算半径
bandwidth = estimate_bandwidth(coordinate, quantile=0.2, n_samples=500)
# bin_seeding: 随便找一些点的中心来作为初始位置
meanShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanShift.fit(coordinate)
labels = meanShift.labels_
plt.scatter(coordinate[:, 0], coordinate[:, 1], c=labels, s=8)
plt.show()
