#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/4 10:25

# @Author : ZFJ

# @File : mnist_dataset.py

# @Software: PyCharm
"""

"""
本模块主要是为了下载Mnist数据集，并将其保存到本地
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(phase='train', batch_size=512, shuffle=True):
    assert phase in ['train', 'test']
    trans = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1037,), (0.3081,))])
    dataset = datasets.MNIST('./mnist_data', train=(phase == 'train'),
                             download=True, transform=trans)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    d = get_dataloader()
    for i in d:
        print(i[0].shape)
