#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: neural_network.py
@time: 2020/9/15 23:35
'''

import torch
import torch.nn as nn


class neural_network(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(neural_network, self).__init__()
        self.w1 = torch.nn.Parameter(torch.FloatTensor(in_dim, 512), requires_grad=True)
        nn.init.kaiming_normal_(self.w1)
        self.b1 = torch.nn.Parameter(torch.FloatTensor(512), requires_grad=True)
        nn.init.zeros_(self.b1)
        self.relu = nn.ReLU(inplace=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(512, out_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.w2)
        self.b2 = torch.nn.Parameter(torch.FloatTensor(out_dim), requires_grad=True)
        nn.init.zeros_(self.b2)
        #self.parameters = nn.ModuleList([self.w1, self.b1, self.w2, self.b2])

    def forward(self, x):
        x = x @ self.w1 + self.b1
        x = self.relu(x)
        x = x @ self.w2 + self.b2
        return x

    # def initialize_weights(self):
    #     nn.init.kaiming_normal_(self.w1)
    #     nn.init.kaiming_normal_(self.w2)
