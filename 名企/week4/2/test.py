#!/usr/bin/env python
# _*_ encoding: utf-8 _*_
"""
@time: 2020/9/9 下午3:20
@file: test.py
@author: caoyuhua
@contact: caoyhseu@126.com
"""

import glob
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import cv2
from FCN_class import FCN8s
from torchvision import transforms


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])  # 直方图均衡化
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


if __name__ == "__main__":
    images_path = './data_road/testing/image_2'
    output_path = './result'
    img_size = (640, 192)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './models/fcn8s_59.pth'
    model = FCN8s(n_class=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for image_name in os.listdir(images_path):
            print(image_name)
            image_path = os.path.join(images_path, image_name)
            image = Image.open(image_path)
            image = image.resize(img_size, Image.BILINEAR)
            image = np.array(image)
            input_tensor = preprocess_image(image)
            output = model(input_tensor)
            output = torch.sigmoid(output)
            pred = torch.argmax(output, dim=1).cpu().numpy().transpose((1, 2, 0))
            pred = np.squeeze(pred)
            image = image[:, :, ::-1]
            image[..., 2] = np.where(pred == 0, 255, image[..., 2])
            cv2.imwrite('{}/{}'.format(output_path, image_name), image)

