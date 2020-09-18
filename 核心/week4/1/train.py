#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author: caoyuhua
@contact: caoyhseu@126.com
@file: train.py
@time: 2020/9/15 22:35
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from neural_network import neural_network
import matplotlib.pyplot as plt
import numpy as np


def draw_figure(train_list, test_list, title="loss", save_path='./figure', img_name="loss.png", x_label='Epoch', y_label="Loss"):
    x = np.arange(1, num_epochs + 1)
    plt.figure()
    plt.plot(x, train_list, color='red', linewidth=1, label='train')
    plt.plot(x, test_list, color='blue', linewidth=1, label='test')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig("{}/{}".format(save_path, img_name))


# 定义超参数
batch_size = 256  # 目前自己编写的hog特征只支持batch_size = 1
learning_rate = 0.001
num_epochs = 50
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
# 加载数据
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=False, transform=transform),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=False, transform=transform),
                         batch_size=batch_size, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = neural_network(784, 10).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss().to(device)

train_acc_list = []
train_loss_list = []
test_acc_list = []
test_loss_list = []

for epoch in range(num_epochs):
    train_loss = 0.0
    num_correct = 0
    for index, (image, label) in enumerate(train_loader):
        #print(len(train_loader.dataset))
        optimizer.zero_grad()
        image = image.view(-1, 28*28).to(device)
        label = label.to(device)

        pred = model(image)

        loss = criterion(pred, label)
        pred = pred.data.max(1)[1]
        # 反向传播
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        num_correct += pred.eq(label.data).sum().item()
    train_acc = num_correct / len(train_loader.dataset)
    train_loss /= len(train_loader)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    print("Train: [Epoch] {}/{}, [Accuracy] {:.2f}%, [Loss] {:.6f}".format(epoch + 1, num_epochs,
                                                                           100 * train_acc, train_loss))
    # 测试阶段
    model.eval()
    test_loss = 0.0
    num_correct = 0

    with torch.no_grad():
        for index, (image, label) in enumerate(test_loader):
            image = image.view(-1, 28*28).to(device)
            label = label.to(device)
            pred = model(image)
            loss = criterion(pred, label)
            test_loss += loss.item()
            pred = pred.data.max(1)[1]
            num_correct += pred.eq(label.data).sum().item()
    test_acc = num_correct / len(test_loader.dataset)
    test_loss /= len(test_loader)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    print("Test: [Epoch] {}/{}, [Accuracy] {:.2f}%, [Loss] {:.6f}".format(epoch + 1, num_epochs,
                                                                           100 * test_acc, test_loss))

# 绘制曲线
draw_figure(train_acc_list, test_acc_list, title='Accuracy', img_name='accuracy.png', y_label='Acc')
draw_figure(train_loss_list, test_loss_list, title='Loss', img_name='loss.png', y_label='Loss')
print("已保存曲线图！！！'")



