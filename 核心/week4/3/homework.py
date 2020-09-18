#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/9 20:16 

# @Author : ZFJ

# @File : homework.py 

# @Software: PyCharm
"""

import torch
import numpy as np
from dataset.mnist_dataset import get_dataloader
from tqdm import tqdm
from skimage import feature as ft
import matplotlib.pyplot as plt

# 设置超参数的值
MAX_EPOCH = 5
LR = 0.001


# 构建绘制函数，用来绘画loss和acc曲线
def draw_figure(train_records, test_records, title="loss", imgname="loss.png", ylabel="Loss"):
    x = np.arange(1, MAX_EPOCH + 1)
    plt.clf()
    plt.plot(x, train_records, color='red', linewidth=1, label='train')
    plt.plot(x, test_records, color='blue', linewidth=1, label='test')
    plt.xticks(np.arange(1, MAX_EPOCH + 1))
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(imgname)


"""
本次实验直接使用skimage.features的hog函数。
因此不必单独计算梯度、幅值（总梯值）和方向。hog函数将在内部计算它并返回特征矩阵。
"""


# 定义hog特征
def get_hog_feature(img_np):
    return ft.hog(img_np,
                  # 方向bin的个数，这里是统计一个胞元内9个方向的梯度直方图
                  orientations=9,
                  # 定义所创建直方图的单元尺寸，原论文中就是8x8方便计算，这里暂时也这么设置
                  pixels_per_cell=(8, 8),
                  # 定义规范化的直方图上的小块区域的大小
                  cells_per_block=(4, 4),
                  # block_norm='L1',
                  # 做一个压缩，很多人也称为伽马校正
                  transform_sqrt=True,
                  # 展平最终的矢量
                  feature_vector=True,
                  # 无需可视化
                  visualize=False)


def main():
    # 载入训练数据
    train_loader = get_dataloader(phase='train', batch_size=1, shuffle=True)
    # 载入验证数据集
    test_loader = get_dataloader(phase='test', batch_size=1, shuffle=False)
    # 构造三层感知机
    mlp_model = torch.nn.Sequential(torch.nn.Linear(in_features=144, out_features=256),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(in_features=256, out_features=10))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp_model.to(device)
    # 使用交叉熵进行评价
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=LR)
    # 最终需要将训练和验证损失和精度，所以这里用来记录
    train_loss_records = []
    train_acc_records = []
    test_loss_records = []
    test_acc_records = []

    for epoch_idx in range(MAX_EPOCH):
        # 训练阶段
        mlp_model.train()
        correct_cnt = 0
        ovr_loss = 0.
        for img, label in tqdm(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            img_np = img.view(32, 32).cpu().numpy()
            feat = get_hog_feature(img_np)
            pred = mlp_model(torch.FloatTensor(feat).to(device))
            loss = criterion(pred.view(1, -1), label)
            loss.backward()
            optimizer.step()
            ovr_loss += loss.item()
            if pred.argmax().item() == label:
                correct_cnt += 1
        train_acc_records.append(correct_cnt / len(train_loader.dataset))
        train_loss_records.append(ovr_loss / len(train_loader.dataset))
        print("Train: [Epoch] {}/{}, [Accuracy] {:.2f}%, [Loss] {:.6f}".format(epoch_idx + 1, MAX_EPOCH,
                                                                               100 * train_acc_records[-1],
                                                                               train_loss_records[-1]))
        # 验证阶段
        mlp_model.eval()
        correct_cnt = 0
        ovr_loss = 0.
        with torch.no_grad():
            for img, label in tqdm(test_loader):
                img, label = img.to(device), label.to(device)
                img_np = img.view(32, 32).cpu().numpy()
                feat = get_hog_feature(img_np)
                pred = mlp_model(torch.FloatTensor(feat).to(device))
                loss = criterion(pred.view(1, -1), label)
                ovr_loss += loss.item()
                if pred.argmax().item() == label:
                    correct_cnt += 1
        test_acc_records.append(correct_cnt / len(test_loader.dataset))
        test_loss_records.append(ovr_loss / len(test_loader.dataset))
        print("Validate: [Epoch] {}/{}, [Accuracy] {:.2f}%, [Loss] {:.6f}".format(epoch_idx + 1, MAX_EPOCH,
                                                                                  100 * test_acc_records[-1],
                                                                                  test_loss_records[-1]))

    # 画图
    draw_figure(train_acc_records, test_acc_records, title='Accuracy', imgname='acc.png', ylabel='Acc')
    draw_figure(train_loss_records, test_loss_records, title='Loss', imgname='loss.png', ylabel='Loss')
    print("保存准确率和Loss曲线图为'acc.png'和'loss.png'")


if __name__ == "__main__":
    main()
