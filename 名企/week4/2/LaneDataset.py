#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@time: 2020/9/9 下午2:32
@file: LaneDataset.py
@author: caoyuhua
@contact: caoyhseu@126.com
"""
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import re
import cv2


class LaneDataset(Dataset):

    ROAD_COLOR = np.array([255, 0, 255])

    def __init__(self, data_dir, img_size=(640, 192)):
        super(LaneDataset, self).__init__()
        self.image_path = [path.replace('\\', '/') for path in glob.glob(os.path.join(data_dir, 'image_2', '*.png'))]
        self.label_path = [os.path.join(data_dir, 'gt_image_2', re.sub('_', '_road_', path.split('/')[-1]))
                           for path in self.image_path]
        self.img_size = img_size

    def __getitem__(self, index):  # 根据索引index返回数据及标签
        path_image = self.image_path[index]
        path_label = self.label_path[index]

        # 图像处理
        image = Image.open(path_image)
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])   # 直方图均衡化
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        image = transforms.ToTensor()(image)

        # 标签处理
        label = Image.open(path_label)
        label = label.resize(self.img_size, Image.NEAREST)
        label = np.array(label)
        gt_image = np.all(label == self.ROAD_COLOR, axis=2)
        gt_image = gt_image.reshape(*gt_image.shape, 1)  # 转为(h,w,1)
        gt_image = np.concatenate((np.invert(gt_image), gt_image), axis=2)
        gt_image = gt_image.transpose([2, 0, 1])
        gt_image = torch.FloatTensor(gt_image)

        return image, gt_image

    def __len__(self):  # 返回整个数据集的大小
        return len(self.image_path)

