#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@time: 2020/9/10 下午2:17
@file: train.py
@author: caoyuhua
@contact: caoyhseu@126.com
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from matplotlib import pyplot as plt
from FCN_class import FCN8s
from LaneDataset import LaneDataset
from metrics import Evaluator


def load_vgg16(model_file):
    model = torchvision.models.vgg16(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


# 绘制曲线
def draw_figures(loss_list, title='loss', save_path='./figure'):
    plt.figure()
    x = np.arange(0, len(loss_list))
    plt.plot(x, loss_list)
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig("{}/{}.png".format(save_path, title))


def main():
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据路径
    train_dir = './data_road/training'
    # 超参数
    num_epochs = 60
    learning_rate = 0.001
    img_size = (640, 192)
    num_class = 2
    SAVE_INTERVAL = 5
    evaluator = Evaluator(num_class=num_class)

    # 构建Dataset实例
    train_data = LaneDataset(data_dir=train_dir, img_size=img_size)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

    # 构建Model实例
    model = FCN8s(n_class=2)
    vgg16_path = "models/vgg16_from_caffe.pth"
    vgg16 = load_vgg16(model_file=vgg16_path)
    model.copy_params_from_vgg16(vgg16)
    model = model.to(device)
    print("模型加载成功！！！")

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    Acc_list = []
    Acc_class_list = []
    mIoU_list = []
    FWIoU_list = []
    loss_list = []

    # 训练
    for epoch in range(num_epochs):
        train_loss = 0.0
        evaluator.reset()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            # 前向传播
            output = model(image)
            output = torch.sigmoid(output)
            loss = criterion(output, label)
            output = torch.argmax(output, dim=1).cpu().numpy()
            label = torch.argmax(label, dim=1).cpu().numpy()

            # 后向传播及优化
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            evaluator.add_batch(label, output)   # 添加output和label，用于后续评估

        # 计算像素准确率、平均像素准确率、平均交并比、频权交并比
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        Acc_list.append(Acc)
        Acc_class_list.append(Acc_class)
        mIoU_list.append(mIoU)
        FWIoU_list.append(FWIoU)
        loss_list.append(train_loss/len(train_loader))

        # 保存模型
        if epoch % SAVE_INTERVAL == 0 or (epoch+1) == num_epochs:
            torch.save(model.state_dict(), './models/fcn8s_{}.pth'.format(epoch))
        print("Epoch_{}: train_loss: {:.6f} Acc: {:.4f} Acc_class: {:.4f} mIoU: {:.4f} FWIoU: {:.4f}"
              .format(epoch+1, train_loss/len(train_loader), Acc, Acc_class, mIoU, FWIoU))

    # 绘制曲线
    draw_figures(loss_list, title='train_loss')
    draw_figures(Acc_list, title='Acc')
    draw_figures(Acc_class_list, title='Acc_class')
    draw_figures(mIoU_list, title='mIoU')
    draw_figures(FWIoU_list, title='FWIoU')
    print("完成曲线绘制！！！")


if __name__ == '__main__':
    main()
