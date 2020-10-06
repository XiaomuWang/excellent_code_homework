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
from loss import FocalLoss, Focal_Loss
from ResnetFcn import ResnetFCN
from resnet import resnet50, resnet34


def load_vgg16(model_file):
    model = torchvision.models.vgg16(pretrained=False)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def load_resnet50(resnet50, model_file):
    #resnet50 = torchvision.models.resnet50(pretrained=False)
    resnet50 = resnet34(pretrained=False)
    state_dict = torch.load(model_file)
    resnet50.load_state_dict(state_dict)
    return resnet50


# 绘制曲线
def draw_figures(train_list, valid_list, title='loss', save_path='./figure', x_label='Epoch', y_label='Loss'):
    plt.figure()
    x = np.arange(1, len(train_list)+1)
    plt.plot(x, train_list, color='red', label='train')
    plt.plot(x, valid_list, color='blue', label='valid')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig("{}/{}.png".format(save_path, title))


def main():
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据路径
    train_dir = './data_road/training'
    valid_dir = './data_road/validing'
    # 超参数
    num_epochs = 20
    learning_rate = 0.001
    img_size = (640, 192)
    num_class = 2
    SAVE_INTERVAL = 5
    evaluator = Evaluator(num_class=num_class)

    # 构建Dataset实例
    train_data = LaneDataset(data_dir=train_dir, img_size=img_size)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    valid_data = LaneDataset(data_dir=valid_dir, img_size=img_size)
    valid_loader = DataLoader(valid_data, batch_size=2, shuffle=False)
    # 构建Model实例
    # model = FCN8s(n_class=2)
    # vgg16_path = "models/vgg16_from_caffe.pth"
    # vgg16 = load_vgg16(model_file=vgg16_path)
    # model.copy_params_from_vgg16(vgg16)

    # resnet101-fcn
    # resnet101_path = './resnet101.pth'
    # resnet101 = torchvision.models.resnet101(pretrained=False)
    # state_dict = torch.load(resnet101_path)
    # resnet101.load_state_dict(state_dict)
    # model = ResnetFCN(resnet101, num_classes=2, expansion=4)

    # resnet50-fcn
    # resnet50_path = './resnet50.pth'
    # resnet50 = torchvision.models.resnet50(pretrained=False)
    # state_dict = torch.load(resnet50_path)
    # resnet50.load_state_dict(state_dict)
    # model = ResnetFCN(resnet50, num_classes=2, expansion=4)

    # resnet34-fcn
    resnet34_path = './resnet34.pth'
    resnet34 = torchvision.models.resnet34(pretrained=False)
    state_dict = torch.load(resnet34_path)
    resnet34.load_state_dict(state_dict)
    model = ResnetFCN(resnet34, num_classes=2, expansion=1)

    model = model.to(device)
    print("模型加载成功！！！")

    # 定义损失函数和优化器
    # criterion = nn.BCELoss()
    criterion = Focal_Loss()
    #criterion = nn.BCELoss(weight=torch.tensor([0.3, 0.7])).to(device)
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75])).to(device)
    #criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    train_Acc_list = []
    train_Acc_class_list = []
    train_mIoU_list = []
    train_FWIoU_list = []
    train_loss_list = []
    valid_Acc_list = []
    valid_Acc_class_list = []
    valid_mIoU_list = []
    valid_FWIoU_list = []
    valid_loss_list = []

    # 训练
    for epoch in range(num_epochs):
        train_loss = 0.0
        evaluator.reset()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            #print(label[:, 1, :, :])
            optimizer.zero_grad()
            # 前向传播
            output = model(image)

            loss = criterion(output, label)
            #output = torch.sigmoid(output)
            output = torch.softmax(output, dim=1)
            #loss = criterion(output.transpose(1,3), label.transpose(1,3))
            output = torch.argmax(output, dim=1).cpu().numpy()
            label = torch.argmax(label, dim=1).cpu().numpy()
            #print(output)

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

        train_Acc_list.append(Acc)
        train_Acc_class_list.append(Acc_class)
        train_mIoU_list.append(mIoU)
        train_FWIoU_list.append(FWIoU)
        train_loss_list.append(train_loss/len(train_loader))

        evaluator.reset()

        # 保存模型
        if (epoch+1) == num_epochs: #epoch % SAVE_INTERVAL == 0 or
            torch.save(model.state_dict(), './models/fcn8s_BFL.pth')
        print("Epoch_{}: train_loss: {:.6f} Acc: {:.4f} Acc_class: {:.4f} mIoU: {:.4f} FWIoU: {:.4f}"
              .format(epoch+1, train_loss/len(train_loader), Acc, Acc_class, mIoU, FWIoU))

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (image, label) in enumerate(valid_loader):
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label)
                #output = torch.sigmoid(output)
                output = torch.softmax(output, dim=1)
                # loss = criterion(output.transpose(1, 3), label.transpose(1, 3))
                output = torch.argmax(output, dim=1).cpu().numpy()
                label = torch.argmax(label, dim=1).cpu().numpy()
                #print(output)
                #print(label)

                valid_loss += loss.item()
                evaluator.add_batch(label, output)   # 添加output和label，用于后续评估

        # 计算像素准确率、平均像素准确率、平均交并比、频权交并比
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        print("Epoch_{}: valid_loss: {:.6f} Acc: {:.4f} Acc_class: {:.4f} mIoU: {:.4f} FWIoU: {:.4f}"
              .format(epoch+1, valid_loss/len(valid_loader), Acc, Acc_class, mIoU, FWIoU))

        valid_Acc_list.append(Acc)
        valid_Acc_class_list.append(Acc_class)
        valid_mIoU_list.append(mIoU)
        valid_FWIoU_list.append(FWIoU)
        valid_loss_list.append(valid_loss/len(valid_loader))

    # 绘制曲线
    draw_figures(train_loss_list, valid_loss_list, title='loss', y_label='loss')
    draw_figures(train_Acc_list, valid_Acc_list, title='Acc', y_label='Acc')
    draw_figures(train_Acc_class_list, valid_Acc_class_list, title='Acc_class', y_label='Acc_class')
    draw_figures(train_mIoU_list, valid_mIoU_list, title='mIoU', y_label='mIoU')
    draw_figures(train_FWIoU_list, valid_FWIoU_list, title='FWIoU', y_label='FWIoU')
    print("完成曲线绘制！！！")


if __name__ == '__main__':
    main()
