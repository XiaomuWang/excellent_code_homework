#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/9 20:00

# @Author : ZFJ

# @File : mnist_dataset.py

# @Software: PyCharm
"""
"""
下载mnist数据集到本地
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloader(phase='train', batch_size=512, shuffle=True):
    assert phase in ['train', 'test']
    trans = transforms.Compose([transforms.Resize(32),
                                transforms.ToTensor()])
    dataset = datasets.MNIST('./mnist_data', train=(phase == 'train'),
                             download=True, transform=trans)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    d = get_dataloader()
    for i in d:
        print(i[0].shape)
