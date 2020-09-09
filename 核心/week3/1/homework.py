#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time : 2020/9/1 21:08 

# @Author : ZFJ

# @File : homework.py 

# @Software: PyCharm
"""
import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt


def generate_data():
    # 本函数生成0-9，10个数字的图片矩阵
    image_data = []
    num_0 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_0)
    num_1 = torch.tensor(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_1)
    num_2 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_2)
    num_3 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_3)
    num_4 = torch.tensor(
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_4)
    num_5 = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_5)
    num_6 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_6)
    num_7 = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0]])
    image_data.append(num_7)
    num_8 = torch.tensor(
        [[0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_8)
    num_9 = torch.tensor(
        [[0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]])
    image_data.append(num_9)
    image_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return image_data, image_label


def get_feature(x):
    # dx
    def s_x(img):
        kernel = np.array([[-1, 0, 1]])
        imgx = signal.convolve2d(img, kernel, boundary='symm', mode='same')
        return imgx

    # dy
    def s_y(img):
        kernel = np.array([[-1, 0, 1]]).T
        imgy = signal.convolve2d(img, kernel, boundary='symm', mode='same')
        return imgy

    # 模长
    def grad(img):
        imgx = s_x(img)
        imgy = s_y(img)
        s = np.sqrt(imgx ** 2 + imgy ** 2)
        theta = np.arctan2(imgx, imgy)
        # 显示角度值
        theta = np.degrees(theta)
        theta[theta < 0] = np.pi + theta[theta < 0]
        return s, theta

    height, width = x.shape
    gradient_magnitude, gradient_angle = grad(x)
    # cell 6*6
    cell_size = 6
    # 360°分8份
    bin_size = 8
    # 分成8份，每一份的度数
    angle_unit = 360 / bin_size
    # 取模长的绝对值
    gradient_magnitude = abs(gradient_magnitude)
    # 整张图片对应多少个cell
    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))

    # 每个cell的特征
    def cell_gradient(cell_magnitude, cell_angle):
        # 建立一个全零的cell的特征向量
        orientation_centers = [0] * bin_size
        # cell是6*6的，遍历每一个像素点
        for k in range(cell_magnitude.shape[0]):
            for l in range(cell_magnitude.shape[1]):
                # 对应的这个像素点的模长
                gradient_strength = cell_magnitude[k][l]
                # 对应的这个像素点的角度
                gradient_angle = cell_angle[k][l]
                # 角度值除以均分的每一部分的度数，整数部分是几就在第几个bin中
                angle = int(gradient_angle / angle_unit)
                # 将模长放入对应的bin中，每个bin的值不断累加更新
                orientation_centers[angle] = orientation_centers[angle] + gradient_strength
        # 返回cell的特征向量
        return orientation_centers

    # 遍历所有cell
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            # 取出图片上对应cell的模长
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            # 取出图片上对应cell的角度
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            # 将所有cell的特征向量串到一起
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

    return cell_gradient_vector


def bin2gray(x):
    # 将二值化的图像转换为灰度图
    gray_img = (x * 255).numpy().astype(np.uint8)
    # 对图像使用最近邻方法做resize, 固定图片大小为256
    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    return gray_img


class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        # 这边将reuqires_grad=True用来标记需要自动求导的位置
        self.weights = torch.nn.Parameter(torch.DoubleTensor(in_dim, out_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.weights)
        self.bias = torch.nn.Parameter(torch.DoubleTensor(out_dim), requires_grad=True)
        nn.init.zeros_(self.bias)
    # 前向计算
    def forward(self, input):
        return torch.mm(input.view(1, -1), self.weights) + self.bias


def train_model(model, data, label, max_epoch=5000, lr=0.001, device='cuda'):
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=lr)
    loss_items = []
    for epoch in range(max_epoch):
        loss = 0
        # 梯度清零，不清零梯度会累加
        for img, lbl in zip(data, label):
            optimizer.zero_grad()
            feat = get_feature(img)
            feat = torch.from_numpy(feat).to(device)
            y_pred = model(feat)
            # l2 distance
            loss += 0.5 * (y_pred - lbl) ** 2
        # 自动计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        print('[Epoch] {}, [Loss] {}'.format(epoch, loss.item()))
        loss_items.append(loss.item())

    # 绘制loss曲线图
    x = np.arange(0, max_epoch)
    plt.plot(x, loss_items, color='red', linewidth=1)

    plt.title("Epoch-Loss Curve")
    plt.savefig("./loss.png")
    print("loss曲线图已保存到loss.png文件")

# 根据特征预测图片
def inference(feat, model, device='cuda'):
    return model(torch.from_numpy(feat).to(device))


if __name__ == "__main__":
    # 先对模型进行训练
    image_data, image_label = generate_data()

    # 放大生成的图片
    image_data = [bin2gray(x) for x in image_data]

    num_sample = len(image_data)
    num_feat = get_feature(image_data[0]).flatten().shape[0]
    learning_rate = 1e-9
    num_epoch = 500
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model = LinearModel(in_dim=num_feat, out_dim=1).to(device)
    train_model(model, image_data, image_label, max_epoch=num_epoch, lr=learning_rate, device=device)

    print("对每张图片进行识别")
    for i in range(0, num_sample):
        x = image_data[i]
        # 对当前图片提取特征
        feature = get_feature(x)
        # 对提取到得特征进行分类
        y = inference(feature, model, device=device)
        # 打印出分类结果
        print("图像[%s]得分类结果是:[%s]" % (i, y.item()))
