#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/15 19:50 

# @Author : ZFJ

# @File : hct66.py 

# @Software: PyCharm
"""
from kmeanspp import *

def generate_data():
    # 本函数生成0-9，10个数字的图片矩阵
    image_data = []
    num_0 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_0)
    num_1 = torch.tensor(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_1)
    num_2 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_2)
    num_3 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_3)
    num_4 = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_4)
    num_5 = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_5)
    num_6 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_6)
    num_7 = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_7)
    num_8 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_8)
    num_9 = torch.tensor(
        [[0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_9)
    image_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return image_data, image_label


def get_feature(x):
    feature = [0, 0, 0, 0]

    # 下面添加提取图像x的特征feature的代码
    def get_shadow(x, dim):
        feature = torch.sum(x, dim)
        feature = feature.float()
        feature = feature.view(1, 6)
        return feature

    feature = get_shadow(x, 0)
    return feature


if __name__ == "__main__":
    image_data, image_label = generate_data()
    # 打印出0的图像
    print("数字0对应的图片是:")
    print(image_data[0])
    print("-" * 20)

    # 打印出8的图像
    print("数字8对应的图片是:")
    print(image_data[8])
    print("-" * 20)

    ITERATION = 100
    kmeanspp = Kmeanspp(3, image_data)
    for i in range(ITERATION):
        kmeanspp.step()

    for i, data in enumerate(image_data):
        clusterType = kmeanspp.getClusterType(data)
        print("数字 [{}] 被分到第 [{}] 类".format(i, clusterType))
