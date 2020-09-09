#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/4 10:55

# @Author : ZFJ

# @File : main.py

# @Software: PyCharm
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataset.mnist_dataset import get_dataloader
from model import get_model
import matplotlib.pyplot as plt
import numpy as np

"""
设置超参数
Batch
Epoch
Learning_rate
开发经验，GPU和CPU选择
"""
BATCH_SIZE = 256
MAX_EPOCH = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义训练模型
def train(net):
    # 读入数据，这边的注意事项就是需要打乱数据
    train_loader = get_dataloader(phase='train', batch_size=BATCH_SIZE, shuffle=True)
    # 选用Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=LR)
    # 选用交叉熵作为损失函数，默认情况下reduction="elementwise_mean",这里选择none输出则是向量
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss_records = []
    net.train()
    for epoch_idx in range(MAX_EPOCH):
        correct_cnt = 0
        ovr_loss = 0.
        for batch_idx, (img, label) in enumerate(train_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            hint = net(img)
            loss = criterion(hint, label)
            loss.mean().backward()
            optimizer.step()
            pred = torch.argmax(hint, dim=-1)
            correct_cnt += ((pred == label).int().sum()).item()
            ovr_loss += loss.sum().item()
        acc = correct_cnt / len(train_loader.dataset)
        mean_loss = ovr_loss / len(train_loader.dataset)
        loss_records.append(mean_loss)
        print('训练 Epoch: {}/{} 准确率: {:.2f}% Loss值: {:.6f}'.
              format(epoch_idx + 1, MAX_EPOCH, 100 * acc, mean_loss))

    return loss_records


# 验证集
def validate(net):
    test_loader = get_dataloader(phase='test', batch_size=1, shuffle=False)
    net.eval()
    with torch.no_grad():
        correct_cnt = 0
        for img, label in test_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            hint = net(img)
            pred = torch.argmax(hint, dim=-1)
            correct_cnt += ((pred == label).int().sum()).item()
    acc = correct_cnt / len(test_loader.dataset)
    print("测试集上的准确率为 {:.2f}".format(acc))


# 绘制最后的损失函数图像
def draw_loss_figure(loss_records, title="loss", imgname="loss.png"):
    x = np.arange(0, len(loss_records))
    plt.plot(x, loss_records, color='red', linewidth=1)
    plt.xticks(np.arange(0, len(loss_records)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(imgname)
    print("loss曲线已保存到{}".format(imgname))
    plt.close()


# 绘制最后的损失函数图像
def draw_loss_figures(loss_records, title="loss"):
    # colors = ['red', 'yellow', 'blue', 'cyan']

    plt.figure()
    for (loss_record, nm) in loss_records:
        x = np.arange(0, len(loss_record))
        plt.plot(x, loss_record, label=nm)
        plt.xticks(np.arange(0, len(loss_record)))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig("All losses.png")
    print("All loss曲线已保存到All losses.png")


if __name__ == "__main__":
    loss_records_list = []
    # 这里作为一个选择，因为以后可能有更多的backbone，那么我们只需要加入到此列表即可
    for backbone_type in ['lenet', 'resnet', 'vgg']:
        print("使用{}网络".format(backbone_type))
        net = get_model(backbone_type).to(DEVICE)
        loss_records = train(net)
        validate(net)
        loss_records_list.append((loss_records, backbone_type))
        draw_loss_figure(loss_records,
                         title="{} loss".format(backbone_type),
                         imgname="{}_loss.png".format(backbone_type))
    draw_loss_figures(loss_records_list, "All losses")
