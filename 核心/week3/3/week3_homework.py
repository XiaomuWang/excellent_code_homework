#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy import signal



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

    def grad(img):
        kernel = np.array([[-1, 0, 1]])
        imgx = signal.convolve2d(img, kernel, boundary='symm', mode='same') 
        kernel = np.array([[-1, 0, 1]]).T
        imgy = signal.convolve2d(img, kernel, boundary='symm', mode='same')
        s = np.sqrt(imgx ** 2 + imgy ** 2)
        theta = np.arctan2(imgx, imgy)
        theta = np.degrees(theta)
        theta[theta < 0] = np.pi + theta[theta < 0]
        return s, theta

    height, width = x.shape
    gradient_magnitude, gradient_angle = grad(x)

    cell_size = 6
    bin_size = 8
    angle_unit = 360 / bin_size
    gradient_magnitude = abs(gradient_magnitude)
    cell_gradient_vector = np.zeros((int(height / cell_size), int(width / cell_size), bin_size))

    #特征
    def cell_gradient(cell_magnitude, cell_angle):
       
        orientation_centers = [0] * bin_size     
        for k in range(cell_magnitude.shape[0]):
            for l in range(cell_magnitude.shape[1]):
               
                gradient_strength = cell_magnitude[k][l]
               
                gradient_angle = cell_angle[k][l]
               
                angle = int(gradient_angle / angle_unit)  

                orientation_centers[angle] = orientation_centers[angle] + gradient_strength
      
        return orientation_centers


    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            
            cell_magnitude = gradient_magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
           
            cell_angle = gradient_angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
    
            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude, cell_angle)

    return cell_gradient_vector


class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
      
        self.weights = torch.nn.Parameter(torch.DoubleTensor(in_dim, out_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.weights)
        self.bias = torch.nn.Parameter(torch.DoubleTensor(out_dim), requires_grad=True)
        nn.init.zeros_(self.bias)
    # 前向计算
    def forward(self, input):
        return torch.mm(input.view(1, -1), self.weights) + self.bias

def train_model(model, data, label, max_epoch=5000, lr=0.001, device='cuda'):
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=lr)

    for epoch in range(max_epoch):
        loss = 0
        # 梯度清零，不清零梯度会累加
        for img, temp in zip(data, label):
            optimizer.zero_grad()
            feat = get_feature(img)
            feat = torch.from_numpy(feat).to(device)
            y_pred = model(feat)
           
            loss += 0.5 * (y_pred - temp) ** 2
        # 自动计算梯度
        loss.backward()

        optimizer.step()
        print('epoch= {}, Loss= {}'.format(epoch, loss.item()))

#灰度图 固定大小为256
def bin2gray(x):

    gray_img = (x * 255).numpy().astype(np.uint8)

    gray_img = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_LINEAR)
    return gray_img

def inference(feat, model, device='cuda'):
    return model(torch.from_numpy(feat).to(device))


if __name__ == "__main__":

    # 获取图片
    image_data, image_label = generate_data()
  
    #图片预处理
    image_data = [bin2gray(x) for x in image_data]

    #特征提取
    img_feature = get_feature(image_data[0]).flatten().shape[0]
    img_sample = len(image_data)

    iters = 200
    learning_rate = 2e-9

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    #模型训练
    model = LinearModel(in_dim=img_feature, out_dim=1).to(device)
    train_model(model, image_data, image_label, max_epoch=iters, lr=learning_rate, device=device)
 
    print("对每张图片进行识别")
    for i in range(0, img_sample):
        x = image_data[i]
        # 对当前图片提取特征
        feature = get_feature(x)
        # 对提取到得特征进行分类
        y = inference(feature, model, device=device)
        # 打印出分类结果
        print("图像[%s]得分类结果是:[%s]" % (i, y.item()))
