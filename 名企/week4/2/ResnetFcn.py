#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: ResnetFcn.py
@time: 2020/9/20 0:34
'''

import torch
import torch.nn as nn
import numpy as np
import torchvision


class ResnetFCN(nn.Module):

    def __init__(self, pretrained_net, num_classes, expansion=4):
        super(ResnetFCN, self).__init__()
        #pretrained_net = torchvision.models.resnet34(pretrained=True)
        self.expansion = expansion
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])  # layer2
        self.stage2 = list(pretrained_net.children())[-4]  # layer3
        self.stage3 = list(pretrained_net.children())[-3]  # layer4

        self.scores1 = nn.Conv2d(512*self.expansion, num_classes, 1)
        self.scores2 = nn.Conv2d(256*self.expansion, num_classes, 1)
        self.scores3 = nn.Conv2d(128*self.expansion, num_classes, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = self.get_upsampling_weight(num_classes, num_classes, 16)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = self.get_upsampling_weight(num_classes, num_classes, 4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = self.get_upsampling_weight(num_classes, num_classes, 4)

        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """双线性插值"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]  # 第一组为纵向产生的kernel_size维数组， 第二组为横向产生的kernel_size维数组
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)  # kernel_zize x kernel_size
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 1/8

        x = self.stage2(x)
        s2 = x  # 1/16

        x = self.stage3(x)
        s3 = x  # 1/32

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)
        s = s1 + s2

        s = self.upsample_8x(s2)
        return s


if __name__ == '__main__':
    resnet50 = torchvision.models.resnet50(pretrained=False, num_classes=2)
    state_dict = torch.load('./resnet50.pth')
    resnet50.load_state_dict(state_dict)
    resfcn = ResnetFCN(resnet50, 2)

