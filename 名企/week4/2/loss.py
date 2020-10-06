#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@time: 2020/9/17 上午8:54
@file: loss.py
@author: caoyuhua
@contact: caoyhseu@126.com
"""
import torch
import torch.nn as nn


# 二分类
class Focal_Loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        pt = torch.softmax(input, dim=1)
        p = pt[:, 1, :, :]
        target = target[:, 1, :, :]
        loss = -self.alpha * (1 - p) ** self.gamma * target * torch.log(p) - \
             (1 - self.alpha) * p ** self.gamma * (1 - target)*torch.log(1 - p)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# 多分类
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
